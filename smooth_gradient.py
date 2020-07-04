import math
import numpy as np
import torch

from saliency_interpreter import SaliencyInterpreter


class SmoothGradient(SaliencyInterpreter):
    """
    Interprets the prediction using SmoothGrad (https://arxiv.org/abs/1706.03825)
    Registered as a `SaliencyInterpreter` with name "smooth-gradient".
    """
    def __init__(self,
                 model,
                 criterion,
                 stdev=0.01,
                 num_samples=10):
        super().__init__(model, criterion)
        # Hyperparameters
        self.stdev = stdev
        self.num_samples = num_samples

    def saliency_interpret(self, test_dataset):

        # Convert inputs to labeled instances
        predictions = self._get_prediction(test_dataset)

        instances_with_grads = dict()
        for idx, (prob, inp, tokens) in enumerate(zip(*predictions)):
            # Run smoothgrad
            label = torch.argmax(prob, axis=0)
            grads = self._smooth_grads(label, inp)

            # Normalize results
            for key, grad in grads.items():
                # TODO (@Eric-Wallace), SmoothGrad is not using times input normalization.
                # Fine for now, but should fix for consistency.

                # The [0] here is undo-ing the batching that happens in get_gradients.
                embedding_grad = np.sum(grad[0], axis=1)
                norm = np.linalg.norm(embedding_grad, ord=1)
                normalized_grad = [math.fabs(e) / norm for e in embedding_grad]
                grads[key] = normalized_grad

            instances_with_grads["instance_" + str(idx + 1)] = grads
            instances_with_grads["instance_" + str(idx + 1)]['tokens_input_1'] = [t[0] for t in tokens]
            instances_with_grads["instance_" + str(idx + 1)]['label_input_1'] = label.item()
            instances_with_grads["instance_" + str(idx + 1)]['prob_input_1'] = prob.max().item()

        return instances_with_grads

    def _register_forward_hook(self, stdev: float):
        """
        Register a forward hook on the embedding layer which adds random noise to every embedding.
        Used for one term in the SmoothGrad sum.
        """

        def forward_hook(module, inputs, output):
            # Random noise = N(0, stdev * (max-min))
            scale = output.detach().max() - output.detach().min()
            noise = torch.randn(output.shape).to(output.device) * stdev * scale

            # Add the random noise
            output.add_(noise)

        # Register the hook
        embedding_layer = self.model.bert.embeddings
        handle = embedding_layer.register_forward_hook(forward_hook)
        return handle

    def _smooth_grads(self, label, inp):
        total_gradients = {}
        for _ in range(self.num_samples):
            handle = self._register_forward_hook(self.stdev)
            grads = self._get_gradients(label, inp)
            handle.remove()

            # Sum gradients
            if total_gradients == {}:
                total_gradients = grads
            else:
                for key in grads.keys():
                    total_gradients[key] += grads[key]

        # Average the gradients
        for key in total_gradients.keys():
            total_gradients[key] /= self.num_samples

        return total_gradients

