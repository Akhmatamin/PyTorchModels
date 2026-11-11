from ui import prediction_page

prediction_page(
    title='Guess a Number',
    emoji='🔢',
    description='Upload a picture of a handwritten digit and the MNIST model will tell you which number it is.',
    endpoint='numbers',
    classes=[str(i) for i in range(10)],
    tips=[
        'Use a **white digit on a black background**, just like the MNIST dataset.',
        'One digit per image, centered and filling most of the frame.',
        'Square images work best — they are resized to 28×28 pixels.',
    ],
)
