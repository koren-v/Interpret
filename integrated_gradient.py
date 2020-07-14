import numpy as np
from tqdm import tqdm

import torch

from saliency_interpreter import SaliencyInterpreter


class IntegratedGradient(SaliencyInterpreter):
    """
    Interprets the prediction using Integrated Gradients (https://arxiv.org/abs/1703.01365)
    Registered as a `SaliencyInterpreter` with name "integrated-gradient".
    """

    def saliency_interpret(self, test_dataloader):

        # Convert inputs to labeled instances
        predictions = self._get_prediction(test_dataloader)

        instances_with_grads = []

        iterator = tqdm(zip(*predictions), total=len(predictions[0]))\
            if self.show_progress else zip(*predictions)

        for probs, inputs, tokens in iterator:

            labels = torch.argmax(probs, dim=1)
            grads = self._integrate_gradients(labels, inputs)

            # sum by embeddings (scalar for each token)
            embedding_grads = grads.sum(dim=2)
            # norm for each sequence
            norms = torch.norm(embedding_grads, dim=1, p=1)
            # normalizing
            for i, norm in enumerate(norms):
                embedding_grads[i] = torch.abs(embedding_grads[i]) / norm

            for i, embedding_grad in enumerate(embedding_grads):
                example_dict = dict()
                # as we do it by batches we has a padding so we need to remove it
                example_tokens = [t for t in tokens[i] if t != '[PAD]']
                example_dict['tokens'] = example_tokens
                example_dict['grad'] = embedding_grad.cpu().tolist()[:len(example_tokens)]
                example_dict['label'] = labels[i].item()
                example_dict['prob'] = probs[i].max().item()
                instances_with_grads.append(example_dict)

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
                embeddings_list.append(output.squeeze(0).clone().detach())

            # Scale the embedding by alpha
            output.mul_(alpha)

        # Register the hook
        embedding_layer = self.model.bert.embeddings
        handle = embedding_layer.register_forward_hook(forward_hook)
        return handle

    def _integrate_gradients(self, label, inp):

        ig_grads = None

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
            if ig_grads is None:
                ig_grads = grads
            else:
                ig_grads = ig_grads + grads

        # Average of each gradient term
        ig_grads /= steps

        # Gradients come back in the reverse order that they were sent into the network
        embeddings_list.reverse()

        # Element-wise multiply average gradient by the input
        ig_grads *= embeddings_list[0]

        return ig_grads
