from ui import prediction_page

prediction_page(
    title='Guess Transport',
    emoji='🚗',
    description='Upload a photo of a vehicle and the model will tell you what kind it is.',
    endpoint='transport',
    classes=['Airplane', 'Bus', 'F1 racing car', 'Motorbike', 'Sedan Car'],
    tips=[
        'Show the whole vehicle, ideally from the side.',
        'Avoid photos with several different vehicles in them.',
    ],
)
