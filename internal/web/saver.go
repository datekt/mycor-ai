package web

import (
	"context"
	"sync"
	"time"
)

/*
saveDebouncer coalesces persistence requests.

The old handler wrote the entire brain to disk inside every /api/train request.
Serialising a large model on each teaching step made the UI unresponsive and
dominated training latency. Instead callers mark the state dirty and this
debouncer performs the actual write:

  - quiet: the write is postponed until no new mark has arrived for `quiet`, so
    a burst of training steps costs exactly one write.
  - max: however many marks keep arriving, a write happens at least every `max`
    interval, so continuous training still reaches disk.
  - stop: flushes any pending write, so shutdown never loses data.

Both timers are armed under the same mutex as the dirty flag, so the state is a
small state machine with no goroutine bookkeeping and no lost wake-ups.
*/
type saveDebouncer struct {
	quiet time.Duration
	max   time.Duration
	save  func()

	mu         sync.Mutex
	dirty      bool
	quietTimer *time.Timer
	maxTimer   *time.Timer
	stopped    bool
	done       chan struct{}
	closeOnce  sync.Once
}

func newSaveDebouncer(quiet, max time.Duration, save func()) *saveDebouncer {
	return &saveDebouncer{
		quiet: quiet,
		max:   max,
		save:  save,
		done:  make(chan struct{}),
	}
}

// mark records that state changed. It never blocks.
func (d *saveDebouncer) mark() {
	d.mu.Lock()
	if d.stopped {
		d.mu.Unlock()
		return
	}
	d.dirty = true

	if d.quietTimer == nil {
		d.quietTimer = time.AfterFunc(d.quiet, d.fire)
	} else {
		d.quietTimer.Reset(d.quiet)
	}
	if d.maxTimer == nil {
		d.maxTimer = time.AfterFunc(d.max, d.fire)
	}
	d.mu.Unlock()
}

// fire performs the write if state is still dirty, and disarms both timers so
// the next mark starts a fresh pair.
func (d *saveDebouncer) fire() {
	d.mu.Lock()
	if d.quietTimer != nil {
		d.quietTimer.Stop()
		d.quietTimer = nil
	}
	if d.maxTimer != nil {
		d.maxTimer.Stop()
		d.maxTimer = nil
	}
	if !d.dirty || d.stopped {
		d.mu.Unlock()
		return
	}
	d.dirty = false
	save := d.save
	d.mu.Unlock()

	if save != nil {
		save()
	}
}

// start launches the watcher that flushes pending state when ctx is cancelled.
func (d *saveDebouncer) start(ctx context.Context) {
	go func() {
		<-ctx.Done()
		d.stop()
	}()
}

// stop stops accepting marks and flushes any pending write. It is safe to call
// more than once.
func (d *saveDebouncer) stop() {
	d.closeOnce.Do(func() {
		close(d.done)
		d.mu.Lock()
		if d.quietTimer != nil {
			d.quietTimer.Stop()
			d.quietTimer = nil
		}
		if d.maxTimer != nil {
			d.maxTimer.Stop()
			d.maxTimer = nil
		}
		pending := d.dirty
		d.dirty = false
		d.stopped = true
		save := d.save
		d.mu.Unlock()

		if pending && save != nil {
			save()
		}
	})
}
