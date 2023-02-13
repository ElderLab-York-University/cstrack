import queue
import threading

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

from pycuda.tools import clear_context_caches


def getInputShape(engine):
    r'''Get input shape of the TensorRT engine.
    '''
    binding = engine[0]
    assert engine.binding_is_input(binding)
    binding_dims = engine.get_binding_shape(binding)

    if len(binding_dims) < 3:
        raise ValueError(f'bad dims of binding {binding}: {binding_dims}')

    return tuple(binding_dims[-2:])


class Executor(threading.Thread):
    r'''Execution context for CUDA tasks.

        Because CUDA data structures are thread-local, in order to ensure consistent
        execution in a multi-thread context, it's necessary to have an execution thread
        through which all CUDA calls are funnelled.
    '''
    def __init__(self):
        r'''Create a new worker thread.
        '''
        super().__init__(target=self.__loop, daemon=True)
        self.__context = None
        self.__input = queue.Queue(1)
        self.__output = queue.Queue(1)
        self.__running = True
        self.start()

    def __del__(self):
        r'''Class destructor.
        '''
        self.stop()

    def __call__(self, task):
        r'''Run a task in the worker thread.
        '''
        self.__input.put(task)
        return self.__output.get()

    def __loop(self):
        r'''Executor thread loop.
        '''
        cuda.init()
        self.__device = cuda.Device(0)
        self.__context = self.__device.make_context()

        while self.__running:
            task = self.__input.get()
            self.__output.put(task())

        self.__context.pop()
        clear_context_caches()

    def stop(self):
        r'''Stop the working thread.
        '''
        self.__running = False
        self.join()


class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple."""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        del self.device
        del self.host


class Output(HostDeviceMem):
    r'''Wrapper for an output buffer.
    '''
    def __init__(self, shape, host_mem, device_mem):
        r'''Create a new output buffer wrapper.
        '''
        super().__init__(host_mem, device_mem)
        self.shape = shape

    @property
    def data(self):
        r'''Return the current output contents.
        '''
        return self.host.reshape(self.shape)


class CSTrack:
    r'''TensorRT-based CSTrack inference.
    '''
    def __init__(self, model_path):
        r'''Create a new CSTrack inference network from a given model file.
        '''
        self.__execute = Executor()

        def start():
            self.trt_logger = trt.Logger(trt.Logger.INFO)
            trt.init_libnvinfer_plugins(self.trt_logger, '')

            with open(model_path, 'rb') as data, trt.Runtime(self.trt_logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(data.read())

            self.context = self.engine.create_execution_context()
            assert self.context is not None

            self.input_shape = getInputShape(self.engine)

            try:
                self.__allocateBuffers()
            except Exception as e:
                raise RuntimeError('failed to allocate CUDA resources') from e

        self.__execute(start)

    def __del__(self):
        r'''Free CUDA memories.
        '''
        def stop():
            del self.outputs
            del self.inputs
            del self.stream

        self.__execute(stop)
        self.__execute.stop()

    def __call__(self, image_fitted, image_original):
        r'''Perform inference on the given image.
        '''
        def detect():
            # Transfer input data to the GPU.
            for (image, inputs) in zip([image_original], self.inputs):
                input_data = self.__preprocess(image)
                inputs.host = np.ascontiguousarray(input_data)
                cuda.memcpy_htod_async(inputs.device, inputs.host, self.stream)

            # Run inference.
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

            # Transfer predictions back from the GPU.
            for outputs in self.outputs:
                cuda.memcpy_dtoh_async(outputs.host, outputs.device, self.stream)

            # Synchronize the stream
            self.stream.synchronize()

            return (self.outputs[1].data, self.outputs[0].data)

        return self.__execute(detect)

    def __allocateBuffers(self):
        r'''Allocates all host/device in/out buffers required for an engine.
        '''
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        output_idx = 0
        for binding in self.engine:
            binding_dims = self.engine.get_binding_shape(binding)
            size = trt.volume(binding_dims)
            #if len(binding_dims) >= 4:
                ## explicit batch case (TensorRT 7+)
                #size = trt.volume(binding_dims)
            #elif len(binding_dims) == 3:
                ## implicit batch case (TensorRT 6 or older)
                #size = trt.volume(binding_dims) * self.engine.max_batch_size
            #else:
                #raise ValueError('bad dims of binding %s: %s' % (binding, str(binding_dims)))

            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer to device bindings.
            self.bindings.append(int(device_mem))

            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                shape = self.engine.get_binding_shape(binding)
                output = Output(shape, host_mem, device_mem)
                print(f'Output layer "{binding} shape = {output.shape}"')
                self.outputs.append(output)
                output_idx += 1

        assert len(self.inputs) == 1, len(self.inputs)
        assert len(self.outputs) == 2, len(self.outputs)

    def __preprocess(self, img):
        """Preprocess an image before TRT YOLO inferencing.

        # Args
            img: int8 numpy array of shape (img_h, img_w, 3)
            input_shape: a tuple of (H, W)
            letter_box: boolean, specifies whether to keep aspect ratio and
                        create a "letterboxed" image for inference

        # Returns
            preprocessed img: float32 numpy array of shape (3, H, W)
        """
        (height, width) = self.input_shape
        img = cv2.resize(img, (width, height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img /= 255.0
        return img
