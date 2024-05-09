from modalities.batch import EvaluationResultBatch
from modalities.messaging.broker.message_broker import MessageBrokerIF
from modalities.messaging.messages.message import BatchProgressUpdate
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
    ) -> MessagePublisher[EvaluationResultBatch]:
        publisher = MessagePublisher[EvaluationResultBatch](
            message_broker=message_broker,
            global_rank=global_rank,
            local_rank=local_rank,
        )
        return publisher

    @staticmethod
    def get_model_state_publisher(
        message_broker: MessageBrokerIF, global_rank: int, local_rank: int
    ) -> MessagePublisher[EvaluationResultBatch]:
        publisher = MessagePublisher[EvaluationResultBatch](
            message_broker=message_broker,
            global_rank=global_rank,
            local_rank=local_rank,
        )
        return publisher
