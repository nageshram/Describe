from flask import Flask,request,render_template,send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from torchvision import transforms
from model import EncoderCNN, DecoderRNN
from nlp_utils import clean_sentence
import os
import pickle
import matplotlib.pyplot as plt

#from werkzeug import secure_filename

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the necessary parameters
embed_size = 256  # Assuming it's the same as during training
hidden_size = 512  # Assuming it's the same as during training
vocab_file = "vocab.pkl"  # Assuming the name of the vocabulary file

# Load the vocabulary
with open(os.path.join("./data", vocab_file), "rb") as f:
    vocab = pickle.load(f)

vocab_size = len(vocab)

# Initialize the encoder and decoder.
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

# Moving models to the appropriate device
encoder.to(device)
decoder.to(device)

# Load the trained weights
encoder_file = "encoder-3.pkl"  # Update with the appropriate file name
decoder_file = "decoder-3.pkl"  # Update with the appropriate file name

encoder.load_state_dict(torch.load(os.path.join("./models", encoder_file)))
decoder.load_state_dict(torch.load(os.path.join("./models", decoder_file)))

# Set models to evaluation mode
encoder.eval()
decoder.eval()


UPLOAD_FOLDER='images'
ALLOWED_EXTENSIONS = ['png','jpg','jpeg']

MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5 MB
MIN_IMAGE_SIZE = 25 * 1024         # 25 KB


app=Flask(__name__)
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

@app.route('/')
def start():
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/images/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method=='POST':
        if 'file' not in request.files:
            return [400,"file not uploaded"]
        file=request.files['file']
        file.seek(0,os.SEEK_END)
        file_size=file.tell()
        file.seek(0)
        if file_size> MAX_IMAGE_SIZE or file_size<MIN_IMAGE_SIZE:
            return [400,"upload file size should be max 5MB and min 2kb"]
        if file.filename=='':
            return [400,"file is empty "]
        if file and allowed_file(file.filename):
            filename=secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            test_image_path=os.path.join(app.config['UPLOAD_FOLDER'],filename) # Replace with the path to your test image
            test_image = Image.open(test_image_path).convert("RGB")

        # Apply transformations to the test image
            transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        # Preprocess the test image
            test_image = transform_test(test_image).unsqueeze(0)  # Add batch dimension

        # Move the preprocessed image to the appropriate device
            test_image = test_image.to(device)

        # Pass the test image through the encoder
            with torch.no_grad():
                features = encoder(test_image).unsqueeze(1)

        # Generate captions with the decoder
            with torch.no_grad():
                output = decoder.sample(features)

        # Convert the output into a clean sentence
            caption = clean_sentence(output, vocab.idx2word)

            return [filename,caption]
        else:
            return [400,"Please upload image file in the format (jpg, png, jpeg)"]


if __name__=='__main__':
    app.run(debug=True)
