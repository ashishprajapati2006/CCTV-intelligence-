"""Kafka Transport Package for Step 10 Event Pipeline.

Exports:
  - KafkaConfig: connection settings
  - KafkaProducerWrapper: producer layer
  - KafkaConsumerWrapper: consumer layer
  - InMemoryKafkaBroker: test/offline broker harness
  - check_kafka_connection: socket ping utility
"""
from src.kafka.client import InMemoryKafkaBroker, check_kafka_connection
from src.kafka.config import KafkaConfig
from src.kafka.consumer import KafkaConsumerWrapper
from src.kafka.producer import KafkaProducerWrapper

__all__ = [
    "KafkaConfig",
    "KafkaProducerWrapper",
    "KafkaConsumerWrapper",
    "InMemoryKafkaBroker",
    "check_kafka_connection",
]
