# SIMDSort

AVX sorting experiment

## Algorithm
![algo](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/algo.svg)  

## Processing Speed

### Uniform Random Values
x4-5 faster than std::sort  
![random](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_random_speed.svg)  

### Imbalanced Random Values (1%: v &in; [0, 1), 99%: v &in; [0, 0.01))
x4-5 faster than std::sort  
![imbalanced random](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_inbalance_speed.svg)  

### NormalDist Random Values
x4-5 faster than std::sort  
![ndist](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_ndist_speed.svg)  

### Reverse Values
disadvantage to quicksort as the number increase  
![ndist](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_reverse_speed.svg)  

### All Conditions
stable speed regardless of conditions
![avxall](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_avxall_speed.svg)  

## Licence
[CC-BY](https://github.com/tk-yoshimura/SIMDSort/blob/main/LICENSE)

## Author

[T.Yoshimura](https://github.com/tk-yoshimura)
