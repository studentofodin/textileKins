import logging
from enum import Enum
from typing import Type
import functools
import time

from omegaconf import DictConfig, OmegaConf
from asyncua.sync import Client, ua, SyncNode, ThreadLoop
from asyncua.client.ua_client import UASocketProtocol

from adanowo_simulator.abstract_base_classes.output_manager import AbstractOutputManager

logger = logging.getLogger(__name__)


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
        self._agentControlState: SyncNode | None = None
        self._agentControlStates = self._create_agent_control_state_enum()

    @property
    def config(self) -> DictConfig:
        return self._config

    @config.setter
    def config(self, c):
        self._config = c

    def _create_agent_control_state_enum(self) -> Type[Enum]:
        state_enum = Enum(
            value="AgentControlState",
            names=[
                ("INVALID", ua.uatypes.Int32(self._config.agent_state_values["invalid"])),
                ("ACCEPTED", ua.uatypes.Int32(self._config.agent_state_values["accepted"])),
                ("REJECTED", ua.uatypes.Int32(self._config.agent_state_values["rejected"]))
            ]
        )
        return state_enum

    def step(self, state: dict[str, float]) -> dict[str, float]:
        if not self._ready:
            raise Exception("Cannot call step() before calling reset().")
        try:
            pass

        except Exception as e:
            self.close()
            raise e
        outputs = {}
        return outputs

    def reset(self, state: dict[str, float]) -> dict[str, float]:
        self.close()

        self._config = self._initial_config.copy()
        self._thread_loop.start()
        self._client = Client(self._config.server_url, tloop=self._thread_loop)

        self._agentControlState = self._get_node_autoconnect(self._config.control_state_node_id)
        # TODO: Define agent input and output nodes

        try:
            self._write_node_autoconnect(self._agentControlState, self._agentControlStates.INVALID.value,
                                         ua.VariantType.Double)
        except Exception as e:
            self.close()
            raise e
        self._ready = True
        # outputs = self.step(state) # TODO: uncomment once step method is implemented
        outputs = {}
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

    @staticmethod
    def _ensure_connection(func):
        @functools.wraps(func)
        def wrapper_ensure_connection(self, *args, **kwargs):
            connection_attempts = 0
            while True:
                try:
                    with self._client:
                        return func(self, *args, **kwargs)
                except (ConnectionError, ua.UaError):
                    time.sleep(config.polling_interval*2)
                    connection_attempts += 1
                    logger.warning(f"Connection error. Trying to reconnect... [{connection_attempts}]")
        return wrapper_ensure_connection

    @_ensure_connection
    def _get_node_autoconnect(self, node_id: dict[str, int]) -> SyncNode:
        return self._client.get_node(ua.NodeId(
            ua.uatypes.Int32(node_id["identifier"]),
            ua.uatypes.Int16(node_id["namespace_index"])
        ))

    @_ensure_connection
    def _write_node_autoconnect(self, node: SyncNode, value: any,
                                datatype: ua.uatypes.VariantType) -> None:
        node.write_value(
            ua.Variant(
                value,
                datatype
            )
        )

    @_ensure_connection
    def read_node_autoconnect(self, node: SyncNode) -> ua.VariantType.Variant:
        return node.read_value()


if __name__ == "__main__":
    config = OmegaConf.load("../config/output_setup/opcua_conn.yaml")
    output_manager = OpcuaOutputManager(config)
    output_manager.reset({})
    # output_manager.step({})
    output_manager.close()