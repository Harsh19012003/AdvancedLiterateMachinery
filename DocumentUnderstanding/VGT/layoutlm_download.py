# Load model directly
from transformers import LayoutLMModel
model = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased")

model.save_pretrained("./weights/layoutlm/")
