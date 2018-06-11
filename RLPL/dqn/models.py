import tensorflow as tf
import tensorflow.contrib.layers as layers


def _mlp(hiddens,
         inpt,
         num_actions,
         scope,
         reuse=False,
         layer_norm=False):
    """
    This model takes an observation as input and returns values of all actions.

    Parameters
    ----------
    hiddens: [int]
        list of sizes of hidden layers

    inpt:
        observation

    num_actions:
        the number of actions

    reuse:
        if reuse


    Returns
    -------
    """
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for hidden in hiddens:
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
            if layer_norm:
                out = layers.layer_norm(out, center=True, scale=True)
            out = tf.nn.relu(out)
        q_out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return q_out


def mlp(hiddens=[], layer_norm=False):
    """
    This is a wrapper of a multilayer perceptron

    Parameters
    ----------
    hiddens: [int]
        list of sizes of hidden layers


    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """
    return lambda *args, **kwargs: _mlp(hiddens, layer_norm=layer_norm, *args, **kwargs)



