Load pytorch models really fast with compression and memory reuse. Achives 3GB/s end to end from S3 to GPU with compression ratios as high as 30% for llama models.

https://developer.download.nvidia.com/compute/nvcomp/3.0/local_installers/nvcomp_3.0.0_x86_64_12.x.tgz

## usage

```
bundle = [self.pipe, self.safety_checker]
# works for any python object with tensors
nyacomp.compress_pickle(bundle, "model/boneless_bundle.pkl")
(self.pipe, self.safety_checker) = nyacomp.load_compressed_pickle("model/boneless_bundle.pkl")
```

This will put compressed tensor files in model/nya and create model/meta.csv for preloading, as well as merged_tensors.csv.

```
model = DiffusionPipeline(...)
# sometimes slower, works for pipelines, state dicts, and modules
nyacomp.compress(model, "model/boneless_model.pth")
nyacomp.load_compressed("model/boneless_model.pth")
```

This will also create model/metadata.pkl if not preloading.

To enable preloading, set PRELOAD_PATH to the location of meta.csv. This will start loading the data in a background thread that releases the GIL, allowing the interpreter to continue with imports. As some imports can be quite slow and other setup work can take time, this can mean that by the time `load_compressed` is called, the data is already ready.

## configuration

* DEBUG: enable debug logging
* SILENT: disable normal logging
* NUM_THREADS: number of threads to use
* NUM_STREAMS: number of cuda streams to use, must be a multiple of NUM_STREAMS

## features

* nyacomp pipelines disk reads or dowloads, GPU transfer, and decompression, using the specified number of threads for parallelizing reads and the specified number of streams for parallelizing transfer and decompression
* a fixed amount of pinned host memory is used (and reused), allowing low CPU memory usage and faster transfer time
* to improve compression ratios and reduce overhead, tensors with identical shapes are merged
* several algorithms are used when distributing work across threads to minimize makespan
* the compressed files and manifest can be downloaded
* a simple Go downloader can be used to buffer downloaded data

## remote downloads

* DOWNLOAD allows for downloading tensors instead of reading them from disk. if DOWNLOAD is set during compression, then HOST may be set to indicate the URL where each tensor should be read. you must manually upload the outputted tensors.
* DOWNLOADER_PATH defaults to /usr/bin/curl, but the remotefile binary may be used instead to avoid pipe backpressure from propagating into TCP backpressure
* if REMOTE_CSV is set, then PRELOAD_PATH may be a url file pointing to a meta.csv file
