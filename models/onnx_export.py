import torch.onnx 
from my_CNN_model import My_CNN_models

#Function to Convert to ONNX 
def Convert_ONNX(model): 
    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(1, 3, 224, 224)  

    # Export the model   
    print("Exporting...")
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "./my_CNN.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
    )
    print(" ") 
    print('Model has been converted to ONNX') 

if __name__ == '__main__':
    model = My_CNN_models(output_dims=50)
    Convert_ONNX(model)