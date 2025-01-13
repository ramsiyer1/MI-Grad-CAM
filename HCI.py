def calculate_hci(ad, ai):
    """
    Calculate the Harmonic Consistency Index (HCI).

    Parameters:
    ad: Normalized Average Drop (AD) score.
    ai: Normalized Average Increase (AI) score.

    Returns:
    hci: Harmonized Confidence Index (HCI).
    """
    if not (0 <= ad <= 1 and 0 <= ai <= 1):
        raise ValueError("AD and AI must be in the range [0, 1].")
    
    numerator = 2 * (1 - ad) * ai
    denominator = (1 - ad) + ai
    
    if denominator == 0:
        raise ValueError("Denominator is zero, resulting in undefined HCI.")
    
    hci = numerator / denominator
    return hci

''' A higher average increase (AI) and a lower average drop (AD) will result in a higher HCI value. Hence a higher HCI signifies a better CAM explainability method. HCI penalizes extreme
    imbalances in both AI and AD and generates a balanced score between AI and AD for a more definitive measure'''
