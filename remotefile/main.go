package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

var chunkSize = 1024 * 1024 // 1MB

var client = &http.Client{}

type DownloadBuffer struct {
	data      [][]byte
	mutex     sync.RWMutex
	cond      *sync.Cond
	done      bool
	chunkPool sync.Pool
}

func NewBuffer() *DownloadBuffer {
	b := &DownloadBuffer{}
	b.cond = sync.NewCond(&b.mutex)
	b.chunkPool = sync.Pool{
		New: func() interface{} {
			return make([]byte, chunkSize)
		},
	}
	return b
}

func (b *DownloadBuffer) Enqueue(chunk []byte) {
	b.mutex.Lock()
	b.data = append(b.data, chunk)
	b.cond.Signal() // Signal to the writer that there's new data
	b.mutex.Unlock()
}

func (b *DownloadBuffer) Dequeue() []byte {
	b.mutex.Lock()
	defer b.mutex.Unlock()
	for len(b.data) == 0 {
		if b.done {
			return nil
		}
		b.cond.Wait() // Wait for new data to be added or downloading to be marked as done
	}
	chunk := b.data[0]
	b.data = b.data[1:]
	return chunk
}

func (b *DownloadBuffer) MarkDone() {
	b.mutex.Lock()
	b.done = true
	b.cond.Signal() // Signal that downloading is done
	b.mutex.Unlock()
}

func (b *DownloadBuffer) IsDone() bool {
	b.mutex.Lock()
	defer b.mutex.Unlock()
	return b.done && len(b.data) == 0
}

func downloadToBuffer(url string, buf *DownloadBuffer) {
	startTime := time.Now()
	resp, err := client.Get(url)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error fetching %s: %v\n", url, err)
		return
	}
	defer resp.Body.Close()

	for {
		chunk := buf.chunkPool.Get().([]byte)
		n, err := resp.Body.Read(chunk)
		if n > 0 {
			buf.Enqueue(chunk[:n]) // Add the data to the buffer
		}
		if err == io.EOF {
			buf.MarkDone()
			elapsed := time.Since(startTime)
			throughput := float64(resp.ContentLength) / elapsed.Seconds() / 1024 / 1024
			fmt.Fprintf(os.Stderr, "Downloaded %s in %s (%.2f MB/s)\n", url, elapsed, throughput)
			buf.chunkPool.Put(chunk)
			return
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading from %s: %v\n", url, err)
			return
		}
	}

}

func writeToStdout(url string, buf *DownloadBuffer) {
	start := time.Now()
	for {
		chunk := buf.Dequeue()
		defer buf.chunkPool.Put(chunk)
		if chunk == nil && buf.IsDone() {
			elapsed := time.Since(start)
			throughput := float64(len(chunk)) / elapsed.Seconds() / 1024 / 1024
			fmt.Fprintf(os.Stderr, "Wrote to stdout in %s (%.2f MB/s)\n", elapsed, throughput)
			return
		}
		// use vmsplice to write chunk to stdout

		// ssize_t vmsplice(int fd, const struct iovec *iov, unsigned long nr_segs, unsigned int flags);

		// 	struct iovec {
		// 		void  *iov_base;        /* Starting address */
		// 		size_t iov_len;         /* Number of bytes */
		// 	};
		if os.Getenv("VMSPLICE") == "1" {
			_, _, err := syscall.Syscall6(syscall.SYS_VMSPLICE, os.Stdout.Fd(), uintptr(unsafe.Pointer(&syscall.Iovec{
				Base: &chunk[0],
				Len:  uint64(len(chunk)),
			})), 1, 0, 0, 0)

			if err != 0 {
				fmt.Fprintf(os.Stderr, "Error splicing %s to stdout: %v\n", url, err)
				return
			}
		} else {
			_, err := os.Stdout.Write(chunk)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error writing %s to stdout: %v\n", url, err)
				return
			}
		}

	}
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "Usage: downloader <url1> <url2> ...")
		os.Exit(1)
	}
	for _, url := range os.Args[1:] {
		// ignore curl args
		if url == "-s" || url == "-v" {
			continue
		}
		buf := NewBuffer()
		// download and write to stdout, but buffer download if stdout is full
		go downloadToBuffer(url, buf)
		writeToStdout(url, buf)
	}
}
