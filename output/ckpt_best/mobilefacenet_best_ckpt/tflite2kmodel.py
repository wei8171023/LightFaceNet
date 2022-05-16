import nncase
import numpy as np

def read_model_file(model_file):
    with open(model_file, 'rb') as f:
        model_content = f.read()
    return model_content

def generate_data(shape, batch):
    shape[0] *= batch
    data = np.random.rand(*shape).astype(np.float32)
    return data

def main():
    model='MobileFaceNet.tflite'
    input_shape=[1,3,112,112]
    # compile_options
    compile_options = nncase.CompileOptions()
    compile_options.target = 'k510'     
    
    
    compile_options.input_type = 'uint8' 
    compile_options.preprocess = True 
    compile_options.input_shape = input_shape
    compile_options.input_layout = 'NCHW'
    compile_options.output_layout = 'NHWC'
    # compile_options.preprocess = True 
    compile_options.mean = [127.5, 127.5, 127.5]
    compile_options.std = [128, 128, 128]
    compile_options.input_range = [0, 255]
    compile_options.dump_ir = True
    compile_options.dump_asm = True
    compile_options.dump_dir = 'nncase_1.6_k510_tmp'
    # compile_options.preprocess = True      # if False, the args below will unworked
    # quantize model
    compile_options.quant_type = 'uint8' # or 'int8'

    # compiler
    compiler = nncase.Compiler(compile_options)

    # import_options
    import_options = nncase.ImportOptions()

    # quantize model
    # compile_options.quant_type = 'uint8' # or 'int8'

    # ptq_options
    ptq_options = nncase.PTQTensorOptions()
    ptq_options.samples_count = 10
    ptq_options.set_tensor_data(generate_data(input_shape, ptq_options.samples_count).tobytes())

    # import
    model_content = read_model_file(model)
    compiler.import_tflite(model_content, import_options)

    # compile
    compiler.use_ptq(ptq_options)
    compiler.compile()

    # kmodel
    kmodel = compiler.gencode_tobytes()
    with open('Mobilefacenet_k510.kmodel', 'wb') as f:
        f.write(kmodel)

if __name__ == '__main__':
    main()
