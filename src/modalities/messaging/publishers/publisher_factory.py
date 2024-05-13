from modalities.messaging.broker.message_broker import MessageBrokerIF
from modalities.messaging.messages.payloads import BatchProgressUpdate, EvaluationResult, StepState
from modalities.messaging.publishers.publisher import MessagePublisher


class PublisherFactory:
    @staticmethod
    def get_batch_progress_publisher(
        message_broker: MessageBrokerIF, global_rank: int, local_rank: int
    ) -> MessagePublisher[BatchProgressUpdate]:
        publisher = MessagePublisher[BatchProgressUpdate](
            message_broker=message_broker,
            global_rank=global_rank,
            local_rank=local_rank,
        )
        return publisher

    @staticmethod
    def get_evaluation_result_publisher(
        message_broker: MessageBrokerIF, global_rank: int, local_rank: int
    ) -> MessagePublisher[EvaluationResult]:
        publisher = MessagePublisher[EvaluationResult](
            message_broker=message_broker,
            global_rank=global_rank,
            local_rank=local_rank,
        )
        return publisher

    @staticmethod
    def get_model_state_publisher(
        message_broker: MessageBrokerIF, global_rank: int, local_rank: int
    ) -> MessagePublisher[StepState]:
        publisher = MessagePublisher[StepState](
            message_broker=message_broker,
            global_rank=global_rank,
            local_rank=local_rank,
        )
        return publisher
