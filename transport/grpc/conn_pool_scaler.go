// Copyright (c) 2026 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package grpc

import (
	"time"

	"go.uber.org/zap"
)

// _scalingMonitorInterval is how often the scaling monitor evaluates the pool.
const _scalingMonitorInterval = 30 * time.Second

// runScalingMonitor runs as a background goroutine for the lifetime of the
// peer.  It periodically evaluates whether connections should be removed
// from the pool.  It exits when the peer's context is cancelled.
func (p *grpcPeer) runScalingMonitor() {
	ticker := time.NewTicker(_scalingMonitorInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			p.evaluateScaling()
		case <-p.ctx.Done():
			return
		}
	}
}

func (p *grpcPeer) evaluateScaling() {
	p.cleanupIdleConns()
	p.maybeScaleDown()
}

// maybeScaleDown checks whether the pool can be reduced by one connection.
// A connection is marked for draining when the remaining active connections
// can absorb the current aggregate stream load without triggering another
// scale-up.
func (p *grpcPeer) maybeScaleDown() {
	p.mu.Lock()
	defer p.mu.Unlock()

	var active []*grpcClientConnWrapper
	for _, c := range p.conns {
		if c.isActive() {
			active = append(active, c)
		}
	}

	// Never drain below minConnections.
	if len(active) <= p.poolCfg.minConnections {
		return
	}

	threshold := int32(float64(p.poolCfg.maxConcurrentStreams) * p.poolCfg.scaleUpThreshold)

	var totalStreams int32
	for _, c := range active {
		totalStreams += c.getStreamCount()
	}

	// Only drain if the remaining (n-1) connections can absorb current load
	// without crossing the scale-up threshold.
	capacityAfterDrain := threshold * int32(len(active)-1)
	if totalStreams >= capacityAfterDrain {
		return
	}

	// Drain the least-loaded active connection.
	var leastLoaded *grpcClientConnWrapper
	for _, c := range active {
		if leastLoaded == nil || c.getStreamCount() < leastLoaded.getStreamCount() {
			leastLoaded = c
		}
	}
	if leastLoaded == nil {
		return
	}

	leastLoaded.setState(connStateDraining)
	p.t.options.logger.Debug("grpc: marked connection for draining during scale-down",
		zap.String("peer", p.address),
		zap.Int32("stream_count", leastLoaded.getStreamCount()))
}

// cleanupIdleConns advances draining connections with zero streams to the idle
// state, and cancels connections that have exceeded the idle timeout so their
// monitor goroutines can close and remove them.
func (p *grpcPeer) cleanupIdleConns() {
	now := time.Now()

	p.mu.Lock()
	var toClose []*grpcClientConnWrapper
	for _, c := range p.conns {
		// Advance draining → idle once all streams have finished.
		if c.getState() == connStateDraining && c.getStreamCount() == 0 {
			c.setState(connStateIdle)
			c.setIdleNow()
		}
		// Collect connections that have been idle past the timeout.
		if c.getState() == connStateIdle && !c.idleSince().IsZero() &&
			now.Sub(c.idleSince()) >= p.poolCfg.idleTimeout {
			toClose = append(toClose, c)
		}
	}
	p.mu.Unlock()

	for _, c := range toClose {
		p.t.options.logger.Debug("grpc: closing idle connection after timeout",
			zap.String("peer", p.address),
			zap.Duration("idle_duration", now.Sub(c.idleSince())))
		// Cancelling the wrapper context causes monitorConnectionStatus to
		// exit, which closes the underlying clientConn and removes the
		// wrapper from p.conns.
		c.cancel()
	}
}
