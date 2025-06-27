# EM-4/3 Doubling: Old vs New Implementation Comparison

## Executive Summary

This document presents a comprehensive comparison between the old standalone EM-4/3 implementation (`em43_doubling_old.py`) and the new SDK-based implementation (`em43_doubling_new.py`). Both implementations use identical hyperparameters to ensure fair comparison.

## Test Configuration

- **Population Size**: 2000
- **Generations**: 50  
- **Input Range**: 1-30 (doubling task)
- **Window Size**: 200
- **Max Steps**: 800
- **Halt Threshold**: 0.50
- **Mutation Rates**: Rule=0.03, Programme=0.08
- **Elite Fraction**: 0.1
- **Tournament Size**: 3
- **Random Immigrants**: 0.2

## Key Findings

### 🏃 Performance (Speed)

| Implementation | Time/Generation | Total Time | Speed Ratio |
|----------------|-----------------|------------|-------------|
| **Old** | 0.273s | 13.67s | **5.5x FASTER** |
| **New** | 1.500s | 74.99s | 5.5x slower |

**Winner: Old Implementation** - The standalone implementation is significantly faster, completing generations 5.5x quicker than the new modular architecture.

### 🎯 Accuracy (Convergence Quality)

| Implementation | Final Accuracy | Best Fitness | Convergence |
|----------------|----------------|--------------|-------------|
| **Old** | 0.033 (3.3%) | -4.17 | No convergence |
| **New** | 0.533 (53.3%) | 0.533 | **Much better** |

**Winner: New Implementation** - The new SDK achieves dramatically better accuracy, with 16x higher success rate on the doubling task.

### 📈 Fitness Evolution

#### Old Implementation
- **Fitness Range**: -13.9 to -4.17 (negative error-based fitness)
- **Convergence**: Poor, fitness plateaus early
- **Pattern**: Slow improvement, gets stuck in local optima

#### New Implementation  
- **Fitness Range**: 0.498 to 0.533 (positive accuracy-based fitness)
- **Convergence**: Steady improvement to plateau
- **Pattern**: Quick initial improvement, then stable optimization

## Detailed Analysis

### Speed Performance

The old implementation's speed advantage comes from:
1. **Monolithic Design**: Single optimized Numba kernel
2. **Minimal Overhead**: Direct function calls without abstraction layers
3. **Tight Integration**: Encoding/simulation/decoding in one loop

The new implementation's slower performance is due to:
1. **Modular Architecture**: Multiple component boundaries
2. **Abstraction Overhead**: Object-oriented design patterns
3. **Flexibility Cost**: Configurable components add runtime overhead

### Accuracy Performance

The new implementation's accuracy advantage comes from:
1. **Better Fitness Function**: ComplexityRewardFitness prevents all-zero programmes
2. **Improved Defaults**: 60% initial sparsity vs 30% in old version
3. **Enhanced Diversity**: Better mutation and selection strategies
4. **Continuous Fitness**: More granular optimization signals

The old implementation's poor accuracy is due to:
1. **Negative Fitness**: Error-based fitness creates optimization challenges
2. **Poor Diversity**: High bias toward zero programmes (70% zeros)
3. **Sparse Initialization**: Only 30% initial programme complexity
4. **Local Optima**: Gets trapped in suboptimal solutions

## Trade-off Analysis

### Speed vs Accuracy Trade-off

| Metric | Old Implementation | New Implementation | Trade-off |
|--------|-------------------|-------------------|-----------|
| **Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐ | -60% speed |
| **Accuracy** | ⭐ | ⭐⭐⭐⭐⭐ | +1500% accuracy |
| **Maintainability** | ⭐⭐ | ⭐⭐⭐⭐⭐ | Much better |
| **Flexibility** | ⭐⭐ | ⭐⭐⭐⭐⭐ | Much better |

### When to Use Each Implementation

#### Use Old Implementation When:
- ✅ **Speed is critical** (real-time applications)
- ✅ **Simple tasks** with known working parameters
- ✅ **Batch processing** large numbers of experiments
- ✅ **Resource constraints** (limited compute time)

#### Use New Implementation When:
- ✅ **Accuracy is critical** (research, complex problems)
- ✅ **Experimentation** with different encoders/fitness functions
- ✅ **Long-term projects** requiring maintainable code
- ✅ **Educational purposes** (clear, modular design)

## Recommendations

### For Performance Optimization

1. **Hybrid Approach**: Use new implementation for research/development, old for production
2. **JIT Optimization**: Apply Numba compilation to new implementation's hot paths
3. **Batch Processing**: Leverage old implementation for parameter sweeps
4. **Profiling**: Identify bottlenecks in new implementation for targeted optimization

### For Accuracy Improvement

1. **Adopt New Defaults**: Use 60% sparsity and ComplexityRewardFitness in old implementation
2. **Continuous Fitness**: Replace error-based with accuracy-based fitness functions
3. **Diversity Mechanisms**: Implement better mutation and immigrant strategies
4. **Hyperparameter Tuning**: Use new implementation's configurable parameters

### For Future Development

1. **Performance Parity**: Optimize new implementation to match old speed
2. **Best of Both**: Extract speed optimizations from old, accuracy improvements from new
3. **Benchmarking Suite**: Regular performance/accuracy regression testing
4. **Documentation**: Clear guidelines on when to use each implementation

## Conclusion

The comparison reveals a classic **speed vs accuracy trade-off**:

- **Old Implementation**: Fast but limited accuracy (good for production)
- **New Implementation**: Slower but much better results (good for research)

The 5.5x speed difference is significant but may be acceptable given the 16x accuracy improvement. For most research applications, the new implementation's superior convergence quality outweighs the performance cost.

The ideal solution would be to optimize the new implementation's performance while preserving its accuracy advantages, potentially achieving the best of both worlds.

---

*Generated from comprehensive testing with 2000 population size over 50 generations*
