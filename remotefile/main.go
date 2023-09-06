// package main

// import (
// 	"fmt"
// 	"io"
// 	"net/http"
// 	"os"
// 	"sync"
// )

// // global HTTP client to reuse connections

// // downloadURL retrieves the content of a given URL using the provided http.Client
// func downloadURL(url string) (io.ReadCloser, error) {
// 	resp, err := client.Get(url)
// 	if err != nil {
// 		return nil, err
// 	}

// 	// Check for a successful HTTP response
// 	if resp.StatusCode != http.StatusOK {
// 		resp.Body.Close()
// 		return nil, fmt.Errorf("failed to download %s, status code: %d", url, resp.StatusCode)
// 	}
// 	return resp.Body, nil
// }

// // writeToStdOut writes data from an io.Reader to stdout, handling potential blockages in the pipe
// func writeToStdOut(reader io.Reader) error {
// 	// unbounded channel of buffers to be written
// 	pending := make(chan []byte)
// 	var wg sync.WaitGroup
// 	wg.Add(1)

// 	// write to stdout in a separate goroutine
// 	go func() {
// 		for buf := range pending {
// 			// Handle potential blockages while writing to stdout
// 			start := 0
// 			for start < len(buf) {
// 				nw, writeErr := os.Stdout.Write(buf[start:])
// 				if writeErr != nil {
// 					panic(writeErr)
// 				}
// 				start += nw
// 			}
// 			// free the buffer
// 			buf = nil
// 		}
// 		wg.Done()
// 	}()

// 	for {
// 		buf := make([]byte, 1024*1024) // 1MB buffer
// 		nr, readErr := reader.Read(buf)
// 		if readErr != nil && readErr != io.EOF {
// 			return readErr
// 		}
// 		if nr == 0 {
// 			break
// 		}
// 		// enqueue the buffer to be written
// 		pending <- buf[:nr]
// 	}
// 	// wait for pending to be empty
// 	close(pending)
// 	wg.Wait()
// 	return nil
// }

// func main() {
// 	if len(os.Args) < 2 {
// 		fmt.Fprintln(os.Stderr, "Usage: downloader <url1> <url2> ...")
// 		os.Exit(1)
// 	}

// 	for _, url := range os.Args[1:] {
// 		body, err := downloadURL(url)
// 		if err != nil {
// 			fmt.Fprintf(os.Stderr, "Error downloading %s: %v\n", url, err)
// 			os.Exit(1)
// 		}
// 		defer body.Close()

// 		if err := writeToStdOut(body); err != nil {
// 			fmt.Fprintf(os.Stderr, "Error writing content of %s to stdout: %v\n", url, err)
// 			os.Exit(1)
// 		}
// 	}
// }

package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"
	"time"
)

var chunkSize = 1024 * 1024 // 1MB

// if os.Getenv("CHUNK_SIZE") {
// 	chunkSize, _ = strconv.Atoi(os.Getenv("CHUNK_SIZE"))
// }

var client = &http.Client{}

type Buffer struct {
	data  [][]byte
	mutex sync.Mutex
	cond  *sync.Cond
	done  bool
}

func NewBuffer() *Buffer {
	b := &Buffer{}
	b.cond = sync.NewCond(&b.mutex)
	return b
}

func (b *Buffer) Enqueue(chunk []byte) {
	b.mutex.Lock()
	b.data = append(b.data, chunk)
	b.cond.Signal() // Signal to the writer that there's new data
	b.mutex.Unlock()
}

func (b *Buffer) Dequeue() []byte {
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

func (b *Buffer) MarkDone() {
	b.mutex.Lock()
	b.done = true
	b.cond.Signal() // Signal that downloading is done
	b.mutex.Unlock()
}

func (b *Buffer) IsDone() bool {
	b.mutex.Lock()
	defer b.mutex.Unlock()
	return b.done && len(b.data) == 0
}

func downloadToBuffer(url string, buf *Buffer) {
	startTime := time.Now()
	resp, err := client.Get(url)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error fetching %s: %v\n", url, err)
		return
	}
	defer resp.Body.Close()

	chunk := make([]byte, chunkSize)
	for {
		n, err := resp.Body.Read(chunk)
		if n > 0 {
			buf.Enqueue(chunk[:n]) // Add the data to the buffer
		}
		if err == io.EOF {
			buf.MarkDone()
			elapsed := time.Since(startTime)
			fmt.Fprintf(os.Stderr, "Downloaded %s in %s\n", url, elapsed)
			return
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading from %s: %v\n", url, err)
			return
		}
	}

}

func writeToStdout(buf *Buffer) {
	start := time.Now()
	for {
		chunk := buf.Dequeue()
		if chunk == nil && buf.IsDone() {
			elapsed := time.Since(start)
			fmt.Fprintf(os.Stderr, "Wrote to stdout in %s\n", elapsed)
			os.Stdout.Close()
			return
		}
		_, err := os.Stdout.Write(chunk)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error writing to stdout: %v\n", err)
			return
		}
	}
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "Usage: downloader <url1> <url2> ...")
		os.Exit(1)
	}

	for _, url := range os.Args[1:] {
		if url == "-s" || url == "-v" {
			continue
		}
		buf := NewBuffer()

		go downloadToBuffer(url, buf)
		writeToStdout(buf)
	}
}