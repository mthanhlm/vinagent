def sale_content(email: str, phone_number: str):
    """
    Sent an apology email to customer
    Args:
        email (str): Customer email to sent apology.
        phone_number (str): Number phone of customer to contact
    Return:
        An apology email content
    """
    content = (
        "Subject: Apology and Resolution for Your Recent Order"
        f"To: {email}; Phone number: {phone_number}\n\n"
        "Dear Valued Customer,\n"
        "I’m very sorry to hear that you received only four of the five items you ordered. That’s not the experience we strive to deliver, and I understand how frustrating this must be.\n"
        "To make things right, here’s what we’d like to do:\n"
        "1. Immediate Replacement\n"
        "– We will ship the missing item at no additional cost to you.\n"
        "– Your replacement will be sent via expedited shipping, on us, so you receive it as quickly as possible.\n"
        "2. Refund Option\n"
        "– If you’d rather receive a refund for the missing item instead of a replacement, please let us know and we’ll process it immediately.\n"
        "3. Goodwill Discount\n"
        "– As an apology for the inconvenience, we’d like to offer you a 15% discount on your next purchase. You can use code SORRY15 at checkout anytime over the next six months.\n\n"
        "Once we have that information, we’ll have the replacement shipped or the refund issued within 24 hours and send you confirmation.\n"
        "Again, I apologize for the trouble and appreciate your patience. Thank you for giving us the chance to make this right.\n\n"
        "Sincerely,\n"
        "Vippro Ecommerce Platform\n"
        "support@vippro_ecm.com | 1-800-123-4567"
    )
    return content
