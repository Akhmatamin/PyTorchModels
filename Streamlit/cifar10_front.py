from ui import prediction_page

prediction_page(
    title='CIFAR-10',
    emoji='🖼️',
    description='Upload a photo and the model will guess which of 10 everyday object classes it shows.',
    endpoint='cifar10',
    classes=['airplane', 'automobile', 'bird', 'cat', 'deer',
             'dog', 'frog', 'horse', 'ship', 'truck'],
    tips=[
        'One main object per photo works best.',
        'Images are shrunk to 32×32 pixels, so keep the object large and centered.',
    ],
)
