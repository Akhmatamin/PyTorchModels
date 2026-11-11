from ui import prediction_page

prediction_page(
    title='Guess a Flower',
    emoji='🌸',
    description='Upload a photo of a flower and the model will tell you which kind it is.',
    endpoint='flowers',
    classes=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'],
    tips=[
        'Get close so the flower head fills most of the photo.',
        'Good lighting helps — avoid very dark or blurry pictures.',
    ],
)
