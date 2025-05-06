.TL
Exercise 2
.AU
Finn Hemsteg
.AI
RWTH Aachen
.ND
.DA "06 May 2025"
.NH
Assignment 
K-means clustering algorithm from scratch
.NH 2
Implementation and Results
.PP
In this exercise, I implemented the K-means clustering algorithm from scratch in Python. The algorithm partitions data into K distinct clusters by iteratively assigning points to the nearest centroid and updating centroids based on cluster means.
.NH 2
Algorithm Overview
.PP
The implementation follows these key steps:
.IP 1.
Initialize K centroids randomly from data points
.IP 2.
Assign each data point to the nearest centroid using Euclidean distance
.IP 3.
Recalculate centroids as the mean of all points in each cluster
.IP 4.
Repeat steps 2-3 until convergence or maximum iterations reached
.PP
To improve results, the algorithm was run 10 times with different random initializations, selecting the run with the lowest Sum of Squared Errors (SSE).
.NH 2
Dataset and Results
.PP
The algorithm was tested on the Iris dataset, with the following results:
.IP \(bu
Converged in 2-11 iterations depending on initialization
.IP \(bu
Best run achieved final SSE: 78.85
.IP \(bu
Optimal centroids:
.br
[5.01, 3.43, 1.46, 0.25]
.br
[5.90, 2.75, 4.39, 1.43]
.br
[6.85, 3.07, 5.74, 2.07]
.NH 2
Visualization
.PP
The plots show:
.IP \(bu
\fIkmeans_clusters.pdf\fR: Final clustering result with K=3, showing clear separation between clusters
.IP \(bu
\fIkmeans_convergence.pdf\fR: Convergence behavior across iterations, demonstrating rapid initial improvement followed by stabilization
.PDFPIC kmeans_clusters.pdf
.PDFPIC kmeans_convergence.pdf
.NH 2
Conclusion
.PP
The K-means implementation successfully identified natural groupings in the Iris dataset. The algorithm demonstrated good convergence properties, typically stabilizing within 6-7 iterations for successful runs. Multiple random initializations proved essential for finding the optimal solution, with the best run achieving an SSE of 78.85.
