import torch

def ema_losses(loss1, loss2, step, x, alpha=0.99):
    """
    Blends loss1 and loss2 using exponential moving average.
    
    Args:
    - loss1: The first loss tensor.
    - loss2: The second loss tensor.
    - step: Current training step.
    - x: Total number of steps for blending.
    - alpha: EMA factor (closer to 1.0 makes the transition smoother).

    Returns:
    - blended_loss: The combined loss.
    """
    # step = 1
    # x = 100
    # ema_factor = 0.99 ** (1 / 100) = 0.0099
    # step =100
    # x = 100
    # ema_factor = 0.99 ** (100 / 100) = 0.99

    if step >= x:
        # Use the final combination after x steps
        return loss2
    else:
        # Blend the losses gradually
        ema_factor = alpha ** (step / x)
        return ema_factor * loss1 + (1 - ema_factor) * (loss2)
    
        # alpha = 0.99
        # step = 1500
        # x = 2000
        # ema_factor = 0.99 ** (1500 / 2000) = 0.817
        # loss = 0.817 * loss1 + 0.183 * loss2

        # step = 1999
        # x = 2000
        # ema_factor = 0.99 ** (1999 / 2000) = 0.99
        # loss = 0.99 * loss1 + 0.01 * loss2

        # step = 2000
        # x = 2000
        # ema_factor = 0.99 ** (2000 / 2000) = 1.0
        # loss = loss2
        


