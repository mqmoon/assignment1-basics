import math

def get_lr_cosine_schedule(t: int, alpha_max: float, alpha_min: float, 
                           Tw: int, Tc: int) -> float:
    if t < Tw:
        return alpha_max * t / Tw
    elif Tw <= t <= Tc:
        progress = (t - Tw) / (Tc - Tw)
        return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + math.cos(progress * math.pi))
    else:
        return alpha_min