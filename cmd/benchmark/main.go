// cmd/benchmark/main.go — LeafcutterLLM Benchmark TUI
//
// Measures the three core performance claims of the Turbo Engine:
//   1. Memory efficiency: peak RAM during layer-by-layer inference
//   2. BLAS speedup: OpenBLAS SGEMM vs naive Go matmul on identical input
//   3. Scheduler throughput: requests/second under concurrent load
//
// All output is a live TUI rendered in the terminal using ANSI escape codes.
// No external TUI libraries are required.

package main

import (
    "context"
    "flag"
    "fmt"
    "math/rand"
    "runtime"
    "sync"
    "sync/atomic"
    "time"

    "github.com/Alartist40/LeafcutterLLM/pkg/qkernel"
    "github.com/Alartist40/LeafcutterLLM/pkg/server"
    "github.com/Alartist40/LeafcutterLLM/pkg/tensor"
)

// ── ANSI helpers ──────────────────────────────────────────────────────────────

const (
    reset  = "\033[0m"
    bold   = "\033[1m"
    green  = "\033[32m"
    yellow = "\033[33m"
    cyan   = "\033[36m"
    red    = "\033[31m"
    dim    = "\033[2m"
)

func header(title string) {
    width := 60
    bar := ""
    for i := 0; i < width; i++ {
        bar += "─"
    }
    fmt.Printf("\n%s%s%s\n", cyan, bar, reset)
    fmt.Printf("%s%s  %s  %s%s\n", cyan, "│", bold+title+reset+cyan, "│", reset)
    fmt.Printf("%s%s%s\n", cyan, bar, reset)
}

func result(label, value, unit string, pass bool) {
    icon := green + "✓" + reset
    if !pass {
        icon = yellow + "~" + reset
    }
    fmt.Printf("  %s  %-38s %s%s %s%s\n", icon, label, bold, value, unit, reset)
}

func divider() {
    fmt.Printf("%s  %-38s%s\n", dim, "────────────────────────────────────", reset)
}

// ── Benchmark 1: Memory efficiency ───────────────────────────────────────────

func benchmarkMemory(hiddenSize, numLayers int) {
    header("BENCHMARK 1 — Memory Efficiency")
    fmt.Printf("  %sSimulating %d transformer layers, hiddenSize=%d%s\n\n",
        dim, numLayers, hiddenSize, reset)

    var baselineStats runtime.MemStats
    runtime.GC()
    runtime.ReadMemStats(&baselineStats)
    baselineAlloc := baselineStats.Alloc

    // Simulate layer-by-layer: only ONE layer's weights in memory at a time.
    layerWeightBytes := hiddenSize * hiddenSize * 4 * 4 // 4 matrices, float32
    peakLayerByLayer := uint64(0)

    for i := 0; i < numLayers; i++ {
        // "Load" layer
        w := make([]float32, hiddenSize*hiddenSize*4)
        for j := range w {
            w[j] = rand.Float32()
        }

        var s runtime.MemStats
        runtime.ReadMemStats(&s)
        current := s.Alloc - baselineAlloc
        if current > peakLayerByLayer {
            peakLayerByLayer = current
        }

        // "Unload" layer
        w = nil
        runtime.GC()
        _ = w
    }

    // Simulate naive: ALL layers in memory at once.
    allWeights := make([][]float32, numLayers)
    for i := range allWeights {
        allWeights[i] = make([]float32, hiddenSize*hiddenSize*4)
    }
    var s runtime.MemStats
    runtime.ReadMemStats(&s)
    naivePeak := s.Alloc - baselineAlloc

    // Clean up
    allWeights = nil
    runtime.GC()

    layerByLayerMB := float64(peakLayerByLayer) / (1024 * 1024)
    naiveMB := float64(naivePeak) / (1024 * 1024)
    theoreticalSingleLayerMB := float64(layerWeightBytes) / (1024 * 1024)
    savings := (1.0 - layerByLayerMB/naiveMB) * 100

    result("Layer-by-layer peak RAM", fmt.Sprintf("%.1f", layerByLayerMB), "MB", true)
    result("Naive (all layers) peak RAM", fmt.Sprintf("%.1f", naiveMB), "MB", false)
    result("Theoretical single layer", fmt.Sprintf("%.1f", theoreticalSingleLayerMB), "MB", true)
    divider()
    result("RAM savings vs naive", fmt.Sprintf("%.1f", savings), "% reduction", savings > 50)
    fmt.Printf("\n  %sConclusion:%s Layer-by-layer uses %s%.1fx less RAM%s than loading all layers.\n",
        dim, reset, bold, naiveMB/layerByLayerMB, reset)
}

// ── Benchmark 2: BLAS vs Naive matmul ────────────────────────────────────────

func benchmarkBLAS(M, N, K, iterations int) {
    header("BENCHMARK 2 — OpenBLAS SGEMM vs Naive Matmul")
    fmt.Printf("  %sMatrix multiply: A[%d×%d] × B[%d×%d]^T, %d iterations%s\n\n",
        dim, M, K, N, K, iterations, reset)

    // Build random Float32 tensors
    A := tensor.NewTensor([]int{M, K}, tensor.Float32)
    B := tensor.NewTensor([]int{N, K}, tensor.Float32)
    aData := A.Data.([]float32)
    bData := B.Data.([]float32)
    for i := range aData {
        aData[i] = rand.Float32()
    }
    for i := range bData {
        bData[i] = rand.Float32()
    }

    // ── BLAS (OpenBLAS SGEMM) ────────────────────────────────────────────────
    var blasTotal time.Duration
    for i := 0; i < iterations; i++ {
        start := time.Now()
        _, err := qkernel.SGEMM(A, B, 1.0, 0.0)
        blasTotal += time.Since(start)
        if err != nil {
            fmt.Printf("  %sOpenBLAS SGEMM error: %v%s\n", red, err, reset)
            return
        }
    }
    blasAvg := blasTotal / time.Duration(iterations)
    blasGFLOPS := float64(2*M*N*K) / float64(blasAvg.Nanoseconds())

    // ── Naive Go triple-loop ─────────────────────────────────────────────────
    // Only run naive on a smaller size to avoid multi-minute waits.
    naiveM, naiveN, naiveK := M, N, K
    naiveIter := iterations
    if M*N*K > 1_000_000 {
        naiveM, naiveN, naiveK = 64, 64, 64
        naiveIter = iterations * 10
        fmt.Printf("  %sNaive test uses smaller matrix [%d×%d] × [%d×%d]^T to avoid timeout%s\n",
            dim, naiveM, naiveK, naiveN, naiveK, reset)
    }

    An := tensor.NewTensor([]int{naiveM, naiveK}, tensor.Float32)
    Bn := tensor.NewTensor([]int{naiveN, naiveK}, tensor.Float32)
    anData := An.Data.([]float32)
    bnData := Bn.Data.([]float32)
    for i := range anData {
        anData[i] = rand.Float32()
    }
    for i := range bnData {
        bnData[i] = rand.Float32()
    }

    var naiveTotal time.Duration
    for i := 0; i < naiveIter; i++ {
        start := time.Now()
        naiveMatmul(An, Bn)
        naiveTotal += time.Since(start)
    }
    naiveAvg := naiveTotal / time.Duration(naiveIter)
    naiveGFLOPS := float64(2*naiveM*naiveN*naiveK) / float64(naiveAvg.Nanoseconds())

    // Extrapolate naive to full size for fair comparison display
    ratio := float64(M*N*K) / float64(naiveM*naiveN*naiveK)
    naiveEquivAvg := time.Duration(float64(naiveAvg) * ratio)
    speedup := float64(naiveEquivAvg) / float64(blasAvg)

    result("OpenBLAS SGEMM avg", blasAvg.String(), "", true)
    result("OpenBLAS throughput", fmt.Sprintf("%.3f", blasGFLOPS), "GFLOPS", true)
    divider()
    result("Naive Go matmul avg (scaled)", naiveEquivAvg.String(), "", false)
    result("Naive Go throughput", fmt.Sprintf("%.3f", naiveGFLOPS), "GFLOPS", false)
    divider()
    result("BLAS speedup", fmt.Sprintf("%.1f", speedup), "x faster", speedup > 2.0)
    fmt.Printf("\n  %sConclusion:%s OpenBLAS is %s%.1fx faster%s than pure Go for this matrix size.\n",
        dim, reset, bold, speedup, reset)
}

// naiveMatmul performs A[M,K] @ B[N,K]^T = C[M,N] with plain Go loops.
func naiveMatmul(A, B *tensor.Tensor) *tensor.Tensor {
    M := A.Shape[0]
    K := A.Shape[1]
    N := B.Shape[0]
    out := tensor.NewTensor([]int{M, N}, tensor.Float32)
    aData := A.Data.([]float32)
    bData := B.Data.([]float32)
    outData := out.Data.([]float32)
    for i := 0; i < M; i++ {
        for j := 0; j < N; j++ {
            var sum float32
            for k := 0; k < K; k++ {
                sum += aData[i*K+k] * bData[j*K+k]
            }
            outData[i*N+j] = sum
        }
    }
    return out
}

// ── Benchmark 3: Scheduler throughput ────────────────────────────────────────

type echoRunner struct {
    processed atomic.Int64
}

func (r *echoRunner) RunBatch(ctx context.Context, batch *server.Batch) error {
    for _, req := range batch.Requests {
        r.processed.Add(1)
        // Simulate ~1ms of inference work per request
        time.Sleep(time.Millisecond)
        req.ResultCh <- server.InferResponse{
            ID:     req.ID,
            Tokens: req.Prompt,
        }
    }
    return nil
}

func benchmarkScheduler(numRequests, batchSize int, waitMs int) {
    header("BENCHMARK 3 — Continuous Batching Scheduler")
    fmt.Printf("  %s%d concurrent requests, batch size=%d, wait=%dms%s\n\n",
        dim, numRequests, batchSize, waitMs, reset)

    runner := &echoRunner{}
    cfg := server.SchedulerConfig{
        MaxBatchSize:    batchSize,
        MaxWaitDuration: time.Duration(waitMs) * time.Millisecond,
        QueueDepth:      numRequests * 2,
    }
    sched := server.NewScheduler(cfg, runner)
    sched.Start()
    defer sched.Stop()

    ctx := context.Background()
    var wg sync.WaitGroup
    latencies := make([]time.Duration, numRequests)
    dropped := atomic.Int64{}

    start := time.Now()

    for i := 0; i < numRequests; i++ {
        wg.Add(1)
        i := i
        go func() {
            defer wg.Done()
            reqStart := time.Now()
            req := &server.InferRequest{
                ID:       fmt.Sprintf("bench-%d", i),
                Prompt:   []int{i % 1000},
                ResultCh: make(chan server.InferResponse, 1),
            }
            _, err := sched.SubmitAndWait(ctx, req)
            if err != nil {
                dropped.Add(1)
                return
            }
            latencies[i] = time.Since(reqStart)
        }()
    }

    wg.Wait()
    totalTime := time.Since(start)

    // Calculate p50 and p99 latency
    var valid []time.Duration
    for _, l := range latencies {
        if l > 0 {
            valid = append(valid, l)
        }
    }
    // Simple sort for percentiles
    for i := 0; i < len(valid)-1; i++ {
        for j := i + 1; j < len(valid); j++ {
            if valid[j] < valid[i] {
                valid[i], valid[j] = valid[j], valid[i]
            }
        }
    }

    var p50, p99 time.Duration
    if len(valid) > 0 {
        p50 = valid[len(valid)*50/100]
        p99 = valid[len(valid)*99/100]
    }

    throughput := float64(runner.processed.Load()) / totalTime.Seconds()

    result("Total requests sent", fmt.Sprintf("%d", numRequests), "", true)
    result("Requests processed", fmt.Sprintf("%d", runner.processed.Load()), "", true)
    result("Requests dropped", fmt.Sprintf("%d", dropped.Load()), "", dropped.Load() == 0)
    divider()
    result("Total time", totalTime.String(), "", true)
    result("Throughput", fmt.Sprintf("%.1f", throughput), "req/sec", throughput > 10)
    result("p50 latency", p50.String(), "", true)
    result("p99 latency", p99.String(), "", true)
    divider()
    result("Batching efficiency", fmt.Sprintf("%.1f", float64(runner.processed.Load())/float64(numRequests)*100), "%", true)
    fmt.Printf("\n  %sConclusion:%s Scheduler processed %s%d/%d requests%s at %.1f req/sec.\n",
        dim, reset, bold, runner.processed.Load(), numRequests, reset, throughput)
}

// ── Main ──────────────────────────────────────────────────────────────────────

func main() {
    hiddenSize  := flag.Int("hidden-size", 512,  "Hidden dimension for memory benchmark")
    numLayers   := flag.Int("num-layers",  32,   "Number of layers to simulate")
    matM        := flag.Int("mat-m",       256,  "Matrix M dimension for BLAS benchmark")
    matN        := flag.Int("mat-n",       256,  "Matrix N dimension for BLAS benchmark")
    matK        := flag.Int("mat-k",       256,  "Matrix K dimension for BLAS benchmark")
    blasIter    := flag.Int("blas-iter",   100,  "BLAS benchmark iterations")
    numReqs     := flag.Int("requests",    200,  "Number of concurrent requests for scheduler benchmark")
    batchSize   := flag.Int("batch-size",  16,   "Scheduler max batch size")
    batchWaitMs := flag.Int("batch-wait",  20,   "Scheduler batch wait in milliseconds")
    flag.Parse()

    width := 60
    bar := ""
    for i := 0; i < width; i++ {
        bar += "═"
    }

    fmt.Printf("\n%s%s%s\n", bold+cyan, bar, reset)
    fmt.Printf("%s  🌿 LeafcutterLLM Turbo Engine — Benchmark Suite  %s\n", bold+cyan, reset)
    fmt.Printf("%s%s%s\n", bold+cyan, bar, reset)
    fmt.Printf("  %sPlatform: %s | CPUs: %d%s\n",
        dim, runtime.GOOS+"/"+runtime.GOARCH, runtime.NumCPU(), reset)

    benchmarkMemory(*hiddenSize, *numLayers)
    benchmarkBLAS(*matM, *matN, *matK, *blasIter)
    benchmarkScheduler(*numReqs, *batchSize, *batchWaitMs)

    width2 := 60
    bar2 := ""
    for i := 0; i < width2; i++ {
        bar2 += "═"
    }
    fmt.Printf("\n%s%s%s\n", bold+cyan, bar2, reset)
    fmt.Printf("%s  Benchmark complete.%s\n", bold, reset)
    fmt.Printf("%s%s%s\n\n", bold+cyan, bar2, reset)
}
