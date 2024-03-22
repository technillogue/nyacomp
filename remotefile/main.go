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

var client *http.Client

func init() {
	client = &http.Client{
		Transport: newTransportWithCacheIP(),
	}
}

type DownloadBuffer struct {
	url       string
	data      [][]byte
	lengths   []int
	mutex     sync.RWMutex
	cond      *sync.Cond
	done      bool
	chunkPool sync.Pool
	// dataChan  chan []byte
}

func NewBuffer(url string) *DownloadBuffer {
	b := &DownloadBuffer{url: url}
	b.cond = sync.NewCond(&b.mutex)
	b.chunkPool = sync.Pool{
		New: func() interface{} {
			return make([]byte, chunkSize)
		},
	}
	return b
}

func (b *DownloadBuffer) Enqueue(chunk []byte, length int) {
	b.mutex.Lock()
	b.data = append(b.data, chunk)
	b.lengths = append(b.lengths, length)
	b.cond.Signal() // Signal to the writer that there's new data
	b.mutex.Unlock()
}

// func (b *DownloadBuffer) actuallySend() {
// 	for {
// 		chunk := <-b.dataChan
// 		if chunk == nil {
// 			return
// 		}
// 		b.mutex.Lock()
// 		b.data = append(b.data, chunk)
// 		b.cond.Signal() // Signal to the writer that there's new data
// 		b.mutex.Unlock()
// 	}
// }

func (b *DownloadBuffer) Dequeue() ([]byte, int) {
	b.mutex.Lock()
	defer b.mutex.Unlock()
	for len(b.data) == 0 {
		if b.done {
			return nil, 0
		}
		b.cond.Wait() // Wait for new data to be added or downloading to be marked as done
	}
	chunk := b.data[0]
	b.data = b.data[1:]
	length := b.lengths[0]
	b.lengths = b.lengths[1:]
	return chunk, length
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

func downloadToBuffer(buf *DownloadBuffer) int64 {
	startTime := time.Now()
	resp, err := client.Get(buf.url)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error fetching %s: %v\n", buf.url, err)
		return 0
	}
	defer resp.Body.Close()

	for {
		chunk := buf.chunkPool.Get().([]byte)
		// memset chunk memory to 0
		for i := range chunk {
			chunk[i] = 0
		}
		n, err := resp.Body.Read(chunk)
		if n > 0 {
			buf.Enqueue(chunk, n) // Add the data to the buffer
		}
		if err == io.EOF {
			buf.MarkDone()
			elapsed := time.Since(startTime)
			throughput := float64(resp.ContentLength) / elapsed.Seconds() / 1024 / 1024
			fmt.Fprintf(os.Stderr, "Downloaded %s in %s (%.2f MB/s)\n", resp.Request.URL, elapsed, throughput)
			buf.chunkPool.Put(chunk)
			return resp.ContentLength
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading from %s: %v\n", resp.Request.URL, err)
			return 0
		}
	}
}

func writeToStdout(buf *DownloadBuffer) {
	totalElapsed := time.Duration(0)
	size := 0
	for {
		chunk, length := buf.Dequeue()
		defer buf.chunkPool.Put(chunk)
		if chunk == nil && buf.IsDone() {
			throughput := float64(size) / totalElapsed.Seconds() / 1024 / 1024
			fmt.Fprintf(os.Stderr, "Wrote %s to stdout in %s (%.2f MB/s)\n", buf.url, totalElapsed, throughput)
			return
		}
		chunkStart := time.Now()
		if os.Getenv("VMSPLICE") == "1" {
			// use vmsplice to write chunk to stdout

			// ssize_t vmsplice(int fd, const struct iovec *iov, unsigned long nr_segs, unsigned int flags);
			// 	struct iovec {
			// 		void  *iov_base;        /* Starting address */
			// 		size_t iov_len;         /* Number of bytes */
			// 	};
			_, _, err := syscall.Syscall6(syscall.SYS_VMSPLICE, os.Stdout.Fd(), uintptr(unsafe.Pointer(&syscall.Iovec{
				Base: &chunk[0],
				Len:  uint64(length),
			})), 1, 0, 0, 0)
			// if we used SPLICE_F_GIFT, we could not reuse the buffer
			// we're not using it, so it's fine, however this might be slower?

			if err != 0 {
				fmt.Fprintf(os.Stderr, "Error splicing %s to stdout: %v\n", buf.url, err)
				return
			}
		} else {
			_, err := os.Stdout.Write(chunk[:length])
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error writing %s to stdout: %v\n", buf.url, err)
				return
			}
		}
		size += len(chunk)
		totalElapsed += time.Since(chunkStart)
	}
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "Usage: downloader <url1> <url2> ...")
		os.Exit(1)
	}
	bufChan := make(chan *DownloadBuffer)
	go func(bufChan chan *DownloadBuffer) {
		total_size := int64(0)
		start := time.Now()
		for _, url := range os.Args[1:] {
			// ignore curl args
			if url == "-s" || url == "-v" {
				continue
			}
			buf := NewBuffer(url)
			bufChan <- buf
			total_size += downloadToBuffer(buf)
		}
		elapsed := time.Since(start)
		throughput := float64(total_size) / elapsed.Seconds() / 1024 / 1024
		fmt.Fprintf(os.Stderr, "Overall downloaded %d MB in %s (%.2f MB/s)\n", total_size/1024/1024, elapsed, throughput)
		close(bufChan)
	}(bufChan)
	for buf := range bufChan {
		writeToStdout(buf)
	}
}
