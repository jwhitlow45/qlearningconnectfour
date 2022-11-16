import visualkeras

from Networks import create_conv2d_model

model = create_conv2d_model()

visualkeras.layered_view(model=model, to_file='ouptut.png')