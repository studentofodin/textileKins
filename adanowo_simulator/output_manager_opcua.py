import logging
from enum import Enum
from typing import Type
import functools
import time
import math

from omegaconf import DictConfig, OmegaConf
from asyncua.sync import Client, ua, SyncNode, ThreadLoop
from asyncua.client.ua_client import UASocketProtocol

from adanowo_simulator.abstract_base_classes.output_manager import AbstractOutputManager

logger = logging.getLogger(__name__)
DIFFERENCE_THRESHOLD = 0.1
INITIAL_USER_FEEDBACK = float(3)  # 3 means user rejected the recommendation, which we assume as default.


class OpcuaOutputManager(AbstractOutputManager):
    """
    OutputManager that uses OPC UA to communicate with the physical environment instead of simulated models.

    Note: This implementation is designed to only use blocking, synchronous operations. This is because the environment
    is slow and the nature of the communication is linear.
    This does not require any concurrency, so we can stay in our happy synchronous world.
    """

    def __init__(self, config: DictConfig):
        # basic config
        self._initial_config: DictConfig = config.copy()
        self._config: DictConfig = self._initial_config.copy()
        self._ready = False
        # network config
        self._thread_loop = ThreadLoop()
        self._client: Client | None = None
        self._agent_control_state_node: SyncNode | None = None
        self._output_nodes: list[SyncNode] | None = None
        self._input_nodes: list[SyncNode] | None = None
        self._agentControlStates = self._create_agent_control_state_enum()

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    def step(self, state: dict[str, float]) -> dict[str, float]:
        if not self._ready:
            raise RuntimeError("Cannot call step() before calling reset().")
        try:
            # each step begins with writing a recommendation to server. Can also be initial state at step 0.
            self._write_recommendation_to_output_nodes(state)
            # indicate that the current recommendation is up to date.
            self._set_agent_control_state("VALID")
            # wait for user decision. Can take much time.
            user_decision = self._await_user_decision()  # in architecture, move "await dead time" to GUI
            #  read process state first after receiving user feedback.
            state_from_server = self._read_agent_inputs(list(state.keys()))
            # indicate that the current recommendation is old end thus invalid.
            self._set_agent_control_state("INVALID")
            # check if the state is plausible. If not, throw warning and use old state.
            only_plausible_states = self._check_state_plausibility(state_from_server)
            # update the internal state with the new state from server.
            self._write_server_state_to_internal_state(state, only_plausible_states)
            # initialize process outputs to be read from server.
            outputs = self._set_outputs_to_initial_values()

            if user_decision == "ACCEPTED":
                # There are only valid measurements in the process outputs if the user accepted the recommendation.
                process_outputs_from_server = self._read_agent_inputs(self._config.output_models)
                only_plausible_process_outputs = self._check_state_plausibility(process_outputs_from_server)
                outputs = self._update_process_outputs(outputs, only_plausible_process_outputs)
                outputs[self._config.user_feedback_key] = float(self._agentControlStates["ACCEPTED"].value)
        except Exception as e:
            self.close()
            raise e
        logger.info("Full step execution successful.")
        return outputs

    def reset(self, state: dict[str, float]) -> dict[str, float]:
        self.close()
        self._config = self._initial_config.copy()
        self._setup_client()

        try:
            self._set_agent_control_state("INVALID")
        except Exception as e:
            self.close()
            raise e
        self._ready = True
        outputs = self.step(state)
        logger.debug("OPC UA connection set up successfully.")
        return outputs

    def close(self) -> None:
        if self._client is not None:
            # check if connection is established
            if (self._client.aio_obj.uaclient.protocol and
                    not self._client.aio_obj.uaclient.state == UASocketProtocol.CLOSED):
                self._client.disconnect()
        if self._thread_loop.is_alive():
            self._thread_loop.stop()
        self._client = None
        self._ready = False

    def _setup_client(self) -> None:
        self._thread_loop.start()
        self._client = Client(self._config.server_url, tloop=self._thread_loop)
        self._agent_control_state_node = self._get_node_autoconnect(self._config.control_state_node_id)
        output_parent_node = self._get_node_autoconnect(self._config.agent_output_node)
        input_parent_node = self._get_node_autoconnect(self._config.agent_input_node)
        self._output_nodes = self._get_node_references_autononnect(output_parent_node)
        self._input_nodes = self._get_node_references_autononnect(input_parent_node)

    def _create_agent_control_state_enum(self) -> Type[Enum]:
        state_enum = Enum(
            value="AgentControlState",
            names=[
                ("INVALID", ua.uatypes.Int64(self._config.agent_state_values["invalid"])),
                ("VALID", ua.uatypes.Int64(self._config.agent_state_values["valid"])),
                ("ACCEPTED", ua.uatypes.Int64(self._config.agent_state_values["accepted"])),
                ("REJECTED", ua.uatypes.Int64(self._config.agent_state_values["rejected"]))
            ]
        )
        return state_enum

    @staticmethod
    def _ensure_connection(func):
        @functools.wraps(func)
        def wrapper_ensure_connection(self, *args, **kwargs):
            connection_attempts = 0
            while True:
                try:
                    with self._client:
                        return func(self, *args, **kwargs)
                except (ConnectionError, ua.UaError) as e:
                    time.sleep(self._config.polling_interval)
                    connection_attempts += 1
                    logger.warning(f"Failed to execute server request: {e.__class__.__name__}: {e}")
                    logger.info(f"Trying to reconnect... [{connection_attempts}]")
                    time.sleep(self._config.polling_interval)
        return wrapper_ensure_connection

    @_ensure_connection
    def _get_node_autoconnect(self, node_id: dict[str, int]) -> SyncNode:
        ns = node_id["namespace_index"]
        i = node_id["identifier"]
        try:
            node = self._client.get_node(
                ua.NodeId(
                    ua.uatypes.Int32(i),
                    ua.uatypes.Int16(ns)
                )
            )
            logger.debug(f"Node retrieval ({node.read_display_name()}) successful.")
        except ua.UaError as e:
            logger.error(f"Node retrieval (ns={ns}, i={i}) failed.")
            raise e
        return node

    @_ensure_connection
    def _write_node_autoconnect(self, node: SyncNode, value: any,
                                datatype: ua.uatypes.VariantType) -> None:
        node.write_value(
            ua.Variant(
                value,
                datatype
            )
        )

    def _set_agent_control_state(self, state: str) -> None:
        self._write_node_autoconnect(self._agent_control_state_node, self._agentControlStates[state].value,
                                     ua.VariantType.Int64)
        logger.debug(f"Successfully set agent control state node to {state}.")

    @_ensure_connection
    def read_node_autoconnect(self, node: SyncNode) -> ua.VariantType.Variant:
        val = node.read_value()
        return val

    @_ensure_connection
    def _get_node_references_autononnect(self, node: SyncNode) -> list[SyncNode]:
        nodes = node.get_referenced_nodes(
            refs=ua.ObjectIds.HasComponent,
            direction=ua.BrowseDirection.Forward,
            nodeclassmask=ua.NodeClass.Variable,
        )
        return nodes

    @_ensure_connection
    def _write_recommendation_to_output_nodes(self, state: dict[str, float]) -> None:
        for node in self._output_nodes:
            node_display_name_str = str(node.read_display_name().Text)
            if node_display_name_str not in state.keys():
                logger.warning(f"Server node {node_display_name_str} not found in state dict.")
                continue
            # Hardcode some exceptions for some setpoints that need to be integers.
            if node_display_name_str == "Cross-lapperLayersCount":
                node.write_value(
                    ua.Variant(
                        int(round(state[node_display_name_str])),
                        ua.VariantType.Int64
                    )
                )
                continue
            node.write_value(
                ua.Variant(
                    state[node_display_name_str],
                    ua.VariantType.Double
                )
            )

    def _await_user_decision(self) -> str:
        while True:
            logger.info(f"Waiting for user decision")
            user_decision_double = ua.uatypes.Int64(self.read_node_autoconnect(self._agent_control_state_node))
            for decision in ["ACCEPTED", "REJECTED"]:
                if user_decision_double == self._agentControlStates[decision].value:
                    logger.info(f"Received user decision: {decision}")
                    return decision
            time.sleep(self._config.polling_interval)

    @_ensure_connection
    def _read_agent_inputs(self, input_names: list[str]) -> dict[str, float]:
        state_from_server = {}
        for node in self._input_nodes:
            node_display_name_str = str(node.read_display_name().Text)
            if node_display_name_str in input_names:
                state_from_server[node_display_name_str] = float(node.read_value())
        if len(state_from_server) < len(input_names):
            missing_states = set(input_names) - set(state_from_server.keys())
            logger.warning(f"Server states missing: {missing_states}")
        return state_from_server

    @staticmethod
    def _check_state_plausibility(state: dict[str, float]) -> dict[str, float]:
        only_plausible_states = {}
        for key, value in state.items():
            if value is None or math.isnan(value):
                logger.warning(f"Not plausible: Server state {key} is None or NaN.")
            elif not isinstance(value, float):
                logger.warning(f"Not plausible: Server state {key} is not a float.")
            elif value < 0:
                logger.warning(f"Not plausible: Server state {key} is negative.")
            else:
                only_plausible_states[key] = value
        return only_plausible_states

    @staticmethod
    def _write_server_state_to_internal_state(state: dict[str, float], server_states: dict[str, float]) -> None:
        """
        Updates the internal state with the new state from server.
        Updates the state by reference (without return value), since state is a mutable object.
        """
        for key, value in server_states.items():
            if abs(state[key] - value) > DIFFERENCE_THRESHOLD:
                logger.warning(f"Server state {key} differs from recommended state "
                               f"by more than {DIFFERENCE_THRESHOLD}. Check if this was intentional.")
            state[key] = value

    def _set_outputs_to_initial_values(self) -> dict[str, float | None]:
        outputs = {}
        for output in self._config.output_models:
            if output in self._config.outputs_always_available:
                outputs[output] = float(0.0)
            else:
                outputs[output] = None
        outputs[self._config.user_feedback_key] = float(self._agentControlStates["REJECTED"].value)

        return outputs

    @staticmethod
    def _update_process_outputs(outputs: dict[str, float | None],
                                process_outputs: dict[str, float]) -> dict[str, float]:
        for key, value in process_outputs.items():
            outputs[key] = value
        return outputs


if __name__ == "__main__":
    test_config = OmegaConf.load("../config/output_setup/opcua_conn.yaml")
    output_manager = OpcuaOutputManager(test_config)
    output_manager.reset({})
    output_manager.close()
