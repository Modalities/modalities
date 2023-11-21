from llm_gym.batch import EvaluationResultBatch
from llm_gym.logging_broker.subscriber import MessageSubscriberIF
from llm_gym.logging_broker.messages import BatchProgressUpdate, Message
import wandb


class WandBEvaluationResultSubscriber(MessageSubscriberIF[EvaluationResultBatch]):
    """A subscriber object for the WandBEvaluationResult observable."""
    def __init__(self, num_ranks: int, project: str, experiment_id: str) -> None:
        super().__init__()
        self.num_ranks = num_ranks
        wandb.init(project=project, name=experiment_id)


    def consume_message(self, message: Message[EvaluationResultBatch]):
        """Consumes a message from a message broker."""
        eval_result = message.payload
        losses = {
            f"{eval_result.dataset_tag} {loss_key}": loss_values for loss_key, loss_values in eval_result.losses.items()
        }
        metrics = {
            f"{eval_result.dataset_tag} {metric_key}": metric_values
            for metric_key, metric_values in eval_result.metrics.items()
        }
        wandb.log(data=losses, step=(eval_result.train_batch_id + 1) * self.num_ranks)
        wandb.log(data=metrics, step=(eval_result.train_batch_id + 1) * self.num_ranks)
