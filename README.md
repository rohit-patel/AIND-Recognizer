# Artificial Intelligence Engineer Nanodegree
## Probabilistic Models
### Completed Project: Sign Language Recognition System

#### Introduction

The project implements hidden markov models for recognizing sign language from videos, given data on hand and nose positions. The solution meets the [rubric here](https://review.udacity.com/#!/rubrics/749/view).

Take a look at the sample [ASL recognizer video](http://www-i6.informatik.rwth-aachen.de/%7Edreuw/download/021.avi) to see how the hand locations are tracked.
#### Summary of results
##### Features and Selectors

1. Ground Features: Left and right hand x-y positions, relative to the nose position and normalized for each speaker.
2. Polar Features: Ground features transformed to polar coordinates.
3. Delta Features: One-period delta of ground features.
4. Custom1 (C1): Difference in left and right hand locations (x and y)
5. Custom2 (C2): Normalized C1

The selectors used were BIC (bayesian information criterion), DIC (discriminative information criterion), and CV (cross validated log likelihood).

##### Results

The scores from various combination of features and selectors were as follows (lower is better).

| Selector | BIC | CV | DIC | Mean | Min |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Polar+Delta+C2 | 0.449438 | 0.477528 | 0.505618 | 0.477528 | 0.449438 |
| Ground+Delta+C2 | 0.449438 | 0.511236 | 0.539326 | 0.500000 | 0.449438 |
| Delta+C1+C2 | 0.455056 | 0.488764 | 0.516854 | 0.486891 | 0.455056 |
| Polar+Delta+C1 | 0.466292 | 0.505618 | 0.516854 | 0.496255 | 0.466292 |
| Ground+Polar+Delta | 0.477528 | 0.595506 | 0.511236 | 0.528090 | 0.477528 |
| Ground+Delta | 0.483146 | 0.505618 | 0.500000 | 0.496255 | 0.483146 |
| Ground+Polar+Delta+C2 | 0.483146 | 0.550562 | 0.522472 | 0.518727 | 0.483146 |
| Ground+Delta+C1 | 0.488764 | 0.578652 | 0.522472 | 0.529963 | 0.488764 |
| Ground+Polar+Delta+C1 | 0.505618 | 0.522472 | 0.567416 | 0.531835 | 0.505618 |
| Ground+Polar+C2 | 0.528090 | 0.522472 | 0.505618 | 0.518727 | 0.505618 |
| Ground+C2 | 0.556180 | 0.539326 | 0.550562 | 0.548689 | 0.539326 |
| Ground+Polar | 0.550562 | 0.567416 | 0.539326 | 0.552434 | 0.539326 |
| Ground | 0.550562 | 0.539326 | 0.573034 | 0.554307 | 0.539326 |
| Ground+C1 | 0.539326 | 0.606742 | 0.539326 | 0.561798 | 0.539326 |
| Polar | 0.544944 | 0.561798 | 0.544944 | 0.550562 | 0.544944 |
| Ground+C1+C2 | 0.567416 | 0.544944 | 0.567416 | 0.559925 | 0.544944 |
| C1+C2 | 0.584270 | 0.623596 | 0.623596 | 0.610487 | 0.584270 |
| C2 | 0.640449 | 0.601124 | 0.662921 | 0.634831 | 0.601124 |
| Delta | 0.612360 | 0.612360 | 0.623596 | 0.616105 | 0.612360 |
| C1 | 0.629213 | 0.646067 | 0.674157 | 0.649813 | 0.629213 |
| Mean | 0.528090 | 0.555056 | 0.555337 | 0.546161 | 0.521910 |

The best performing combination was **Polar + Delta + C1 with a BIC selector**.
