package main

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
)

const (
	ord1RegionKey = "ORD1"
	las1RegionKey = "LAS1"
	lga1RegionKey = "LGA1"
)

var regionToCacheURL = map[string]string{
	ord1RegionKey: "storagecache-level1-ord1.tenant-replicate-prdsvcs.svc.cluster.local",
	las1RegionKey: "storagecache-level1-las1.tenant-replicate-prdsvcs.svc.cluster.local",
	lga1RegionKey: "storagecache-level1-lga1.tenant-replicate-prdsvcs.svc.cluster.local",
}

var cidrToRegion = map[string]string{
	// https://replicatehq.slack.com/archives/C05931N39B7/p1697478414882239?thread_ts=1697475428.371909&cid=C05931N39B7
	"10.131.0.0/16": las1RegionKey,
	"10.134.0.0/16": ord1RegionKey,
	"10.135.0.0/16": ord1RegionKey,
	"10.137.0.0/16": lga1RegionKey,
	"10.161.0.0/16": las1RegionKey,
	"10.165.0.0/16": las1RegionKey,
}

func getCWRegion(addr string) (string, error) {
	nodeAddr := net.ParseIP(addr)
	if nodeAddr == nil {
		return "", fmt.Errorf("failure to parse ip address from NODE_IP env: %s", addr)
	}
	for cidrBlock, region := range cidrToRegion {
		_, cidr, err := net.ParseCIDR(cidrBlock)
		if err != nil {
			return "", err
		}
		if cidr.Contains(nodeAddr) {
			return region, nil
		}
	}
	return "", errors.New("no matching CIDR block found")
}

func getCacheIP() (string, error) {
	nodeIP := os.Getenv("PGET_HOST_IP")
	if nodeIP == "" {
		return "", errors.New("PGET_HOST_IP env var not set")
	}
	region, err := getCWRegion(nodeIP)
	if err != nil {
		return "", err
	}
	cacheURL := regionToCacheURL[region]
	addrs, err := net.LookupIP(cacheURL)
	if err != nil {
		return "", fmt.Errorf("failed to lookup service IP for %s: %w", cacheURL, err)
	}
	if len(addrs) == 0 {
		return "", fmt.Errorf("no IP addresses found for %s", cacheURL)
	}
	fmt.Fprintf(os.Stderr, "Remotefile using cache IP %s for region %s\n", addrs[0].String(), region)
	return addrs[0].String(), nil
}

func newTransportWithCacheIP() *http.Transport {
	cacheIP, err := getCacheIP()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to get cache IP, not using custom dialer: %s\n", err)
		return &http.Transport{MaxConnsPerHost: 100}
	}
	hostname := "weights.replicate.delivery:443"
	addrWithPortMap := map[string]string{
		hostname + ":443": cacheIP + ":443",
		hostname + ":80":  cacheIP + ":80",
	}
	return &http.Transport{
		MaxConnsPerHost: 100,
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			if customAddr, ok := addrWithPortMap[addr]; ok {
				addr = customAddr
			}
			return net.Dial(network, addr)
		},
	}
}
