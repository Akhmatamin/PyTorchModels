from ui import prediction_page

prediction_page(
    title='Guess Fashion',
    emoji='👕',
    description='Upload a photo of a clothing item and the Fashion-MNIST model will name it.',
    endpoint='fashion',
    classes=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
             'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
    tips=[
        'Show a single item on a plain, dark background.',
        'Make the item fill most of the picture.',
        'Simple product-style photos work better than people wearing the clothes.',
    ],
)
