import math
import numpy as np

from saliency_interpreter import SaliencyInterpreter


class IntegratedGradient(SaliencyInterpreter):
    """
    Interprets the prediction using Integrated Gradients (https://arxiv.org/abs/1703.01365)
    Registered as a `SaliencyInterpreter` with name "integrated-gradient".
    """

    def saliency_interpret(self, test_dataset):

        # Convert inputs to labeled instances
        predictions = self._get_prediction(test_dataset)

        instances_with_grads = dict()
        for idx, (label, inp, tokens) in enumerate(zip(*predictions)):
            # Run smoothgrad
            grads = self._integrate_gradients(label, inp)

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

        return instances_with_grads

    def _register_forward_hook(self, alpha, embeddings_list):
        """
        Register a forward hook on the embedding layer which scales the embeddings by alpha. Used
        for one term in the Integrated Gradients sum.
        We store the embedding output into the embeddings_list when alpha is zero.  This is used
        later to element-wise multiply the input by the averaged gradients.
        """

        def forward_hook(module, inputs, output):
            # Save the input for later use. Only do so on first call.
            if alpha == 0:
                embeddings_list.append(output.squeeze(0).clone().detach().numpy())

            # Scale the embedding by alpha
            output.mul_(alpha)

        # Register the hook
        embedding_layer = self.model.bert.embeddings
        handle = embedding_layer.register_forward_hook(forward_hook)
        return handle

    def _integrate_gradients(self, label, inp):

        ig_grads = {}

        # List of Embedding inputs
        embeddings_list = []

        # Use 10 terms in the summation approximation of the integral in integrated grad
        steps = 10

        # Exclude the endpoint because we do a left point integral approximation
        for alpha in np.linspace(0, 1.0, num=steps, endpoint=False):
            # Hook for modifying embedding value
            handle = self._register_forward_hook(alpha, embeddings_list)

            grads = self._get_gradients(label, inp)
            handle.remove()

            # Running sum of gradients
            if ig_grads == {}:
                ig_grads = grads
            else:
                for key in grads.keys():
                    ig_grads[key] += grads[key]

        # Average of each gradient term
        for key in ig_grads.keys():
            ig_grads[key] /= steps

        # Gradients come back in the reverse order that they were sent into the network
        embeddings_list.reverse()

        # Element-wise multiply average gradient by the input
        for idx, input_embedding in enumerate(embeddings_list):
            key = "grad_input_" + str(idx + 1)
            ig_grads[key] *= input_embedding

        return ig_grads
