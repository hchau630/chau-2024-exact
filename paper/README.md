# Commands for reproducing figures

| Figure | Command (run in the enclosing directory) |
|--------|------------------------------------------|
| 1B | `python paper/fit_kernel.py paper/product_data/EI.csv paper/product_data/IE.csv -o paper/figures/1b.pdf` |
| 1D (left) | `python paper/lineplot.py compare -d 2 -s 125 90 85 110 --wee 3 --wei 4 --wie 4 --wii 5.25 --kee 0.5 --kei -0.25 --kie -0.25 --kii 0.25 --N-ori 12 --N-osi 7 --approx-order 3 -o paper/figures/1d_a.pdf` |
| 1D (right) | `python paper/lineplot.py compare -d 2 -s 125 90 85 110 --wee 3 --wei 4 --wie 4 --wii 5.25 --kee 0.5 --kei -0.25 --kie -0.25 --kii 0.25 --N-ori 12 --N-osi 7 --approx-order 3 -m ori_osi -o paper/figures/1d_b.pdf` |
| 2A | `python paper/contourplot.py space_2 --rho 1 -o paper/figures/2a.pdf` |
| 2B | `python paper/contourplot.py space_2 --rho 1 2 -o paper/figures/2b.pdf` |
| 2C | `python paper/lineplot.py response --wee 4.5 1.5 1.5 --wei 3 3 2 --wie 2 2.5 2.5 --wii 0 4 4 -s 40 40 --axsize 1.25 1 -o paper/figures/2c.pdf` |
| 2D | `python paper/contourplot.py space_zero -x -5 5 -y 0 20 --w00 5 --rho 0.72 -c E -l -1.25 1 -n 10 -N 100 -d 2 --shade 0.547 0.443 0.691 -o paper/figures/2d.pdf` |
| 2E | `python paper/contourplot.py space_zero -x -5 5 -y 0 20 -c E -l -1.25 1.25 -d 2 -N 100 --mode min_diff --w00 5 --rho 0.72 -n 9 -l 0 2 -s linear --shade 0.370 0.260 0.530 -o paper/figures/2e.pdf` |
| 2F | `python paper/contourplot.py space_zero -m diff -x -5 5 -y 0 20 --w00 5 --rho 0.72 -c E -l -1.25 1.25 -l 0 1.5 -n 7 -d 2 -N 100 -o paper/figures/2f.pdf` |
| 2G | `python paper/contourplot.py space_2 -m decay --rho 0.72 -o paper/figures/2g.pdf` |
| 3A | `python paper/contourplot.py space_2 --rho 0.72 -c EI -y 0 20 -x -7.5 5 -o paper/figures/3a.pdf` |
| 3B | `python paper/contourplot.py space_zero -d 2 -N 100 --w00 5 --rho 0.72 -m diff_EI -c E -x -7.5 5 -y 0 20 -l -0.5 1 -s halflog --linthresh 2 -n 7 --axsize 2.5 1.85 -o paper/figures/3b.pdf` |
| 4A | `python paper/lineplot.py response -m ori --wee 0.5 0.5 --wei -0.5 -1 --wie 0.5 1 --wii 0 0 --no-normalize -o paper/figures/4a.pdf` |
| 4B | `python paper/contourplot.py mean --axes ori -x 0 2 -y -2 2 -N 500 -o paper/figures/4b.pdf` |
| 4C | `python paper/lineplot.py response -d 2 -s 100 100 --wee 1.5 0.2 1.5 0.2 --wei -3 -1 -3 -1 --wie 3 0.9 3 0.45 --wii -5 -2.5 -5 -2.5 -m space_ori --axsize 1.25 1.4 -o paper/figures/4c.pdf` |
| 4D | `python paper/contourplot.py space_ori -x -2 2 -y -2 2 --rho 0.72 --w00 0.2 -c E --axsize 2.25 2 -N 1000 -o paper/figures/4d_alt.pdf` |
| 5A | `python paper/mean_response_gain.py -o paper/figures/5a.pdf` |
| 5B | Same as Figure 2B
| 5C | `python paper/contourplot.py space_zero -x -5 5 -y 0 20 --w00 5 -l -0.5 1 -n 7 -s neghalflog -m gain --rho 0.72 -c E -d 2 -N 100 --axsize 2.5 1.85 -o paper/figures/5c.pdf` |
| 5D | Same as Figure 4B |
| 6 | See `paper/model_fits/README.md` |
| S1 (1st row) | `python paper/contourplot.py space_zero -x -5 5 -y 0 20 --w00 1 --rho 0.8 1 1.25 -c E -l -4 1 -n 21 -N 100 -d 2 --shade 0.547 0.443 0.691 -o paper/figures/supp/S1_a.pdf` |
| S1 (2nd row) | `python paper/contourplot.py space_zero -x -5 5 -y 0 20 --w00 2.5 --rho 0.8 1 1.25 -c E -l -1.25 1 -n 10 -N 100 -d 2 --shade 0.547 0.443 0.691 -o paper/figures/supp/S1_b.pdf` |
| S1 (3rd row) | `python paper/contourplot.py space_zero -x -5 5 -y 0 20 --w00 5 --rho 0.8 1 1.25 -c E -l -1.25 1 -n 10 -N 100 -d 2 --shade 0.547 0.443 0.691 -o paper/figures/supp/S1_c.pdf` |
| S2 (1st row) | `python paper/contourplot.py space_zero -x -5 5 -y 0 20 -c E -l -1.25 1.25 -d 2 -N 100 --mode min_diff --w00 1 --rho 0.8 1 1.25 -n 9 -l 0 2 -s linear --shade 0.370 0.260 0.530 -o paper/figures/supp/S2_a.pdf` |
| S2 (2nd row) | `python paper/contourplot.py space_zero -x -5 5 -y 0 20 -c E -l -1.25 1.25 -d 2 -N 100 --mode min_diff --w00 2.5 --rho 0.8 1 1.25 -n 9 -l 0 2 -s linear --shade 0.370 0.260 0.530 -o paper/figures/supp/S2_b.pdf` |
| S2 (3rd row) | `python paper/contourplot.py space_zero -x -5 5 -y 0 20 -c E -l -1.25 1.25 -d 2 -N 100 --mode min_diff --w00 5 --rho 0.8 1 1.25 -n 9 -l 0 2 -s linear --shade 0.370 0.260 0.530 -o paper/figures/supp/S2_c.pdf` |
| S3 (1st row) | `python paper/contourplot.py space_zero -m diff -x -5 5 -y 0 20 --w00 1 --rho 0.8 1 1.25 -c E -l -1.25 1.25 -l 0 1.5 -n 7 -d 2 -N 100 -o paper/figures/supp/S3_a.pdf` |
| S3 (2nd row) | `python paper/contourplot.py space_zero -m diff -x -5 5 -y 0 20 --w00 2.5 --rho 0.8 1 1.25 -c E -l -1.25 1.25 -l 0 1.5 -n 7 -d 2 -N 100 -o paper/figures/supp/S3_b.pdf` |
| S3 (3rd row) | `python paper/contourplot.py space_zero -m diff -x -5 5 -y 0 20 --w00 5 --rho 0.8 1 1.25 -c E -l -1.25 1.25 -l 0 1.5 -n 7 -d 2 -N 100 -o paper/figures/supp/S3_c.pdf` |
| S4 (1st row) | `python paper/contourplot.py space_zero -d 2 -N 100 --w00 1 -m diff_EI -c E -x -7.5 5 -y 0 20 -l -1 1 -s symlog -n 5 --rho 0.8 1 1.25 -t 0.2 -o paper/figures/supp/S4_a.pdf` |
| S4 (2nd row) | `python paper/contourplot.py space_zero -d 2 -N 100 --w00 2.5 -m diff_EI -c E -x -7.5 5 -y 0 20 -l -1 1 -s symlog -n 5 --rho 0.8 1 1.25 -t 0.2 -o paper/figures/supp/S4_b.pdf` |
| S4 (3rd row) | `python paper/contourplot.py space_zero -d 2 -N 100 --w00 5 -m diff_EI -c E -x -7.5 5 -y 0 20 -l -1 1 -s symlog -n 5 --rho 0.8 1 1.25 -t 0.2 -o paper/figures/supp/S4_c.pdf` |
| S5 (1st row) | `python paper/contourplot.py space_zero -x -5 2.5 -y 0 5 --w00 1 -l -1 1 -s symlog -n 11 -m gain --rho 0.8 1 1.25 -c E -d 2 -N 100 --axsize 2.5 2 -o paper/figures/supp/S5_a.pdf` |
| S5 (2nd row) | `python paper/contourplot.py space_zero -x -5 2.5 -y 0 5 --w00 2.5 -l -1 1 -s symlog -n 11 -m gain --rho 0.8 1 1.25 -c E -d 2 -N 100 --axsize 2.5 2 -o paper/figures/supp/S5_b.pdf` |
| S5 (3rd row) | `python paper/contourplot.py space_zero -x -5 2.5 -y 0 5 --w00 5 -l -1 1 -s symlog -n 11 -m gain --rho 0.8 1 1.25 -c E -d 2 -N 100 --axsize 2.5 2 -o paper/figures/supp/S5_c.pdf` |

# Additional commands
To get the data files `paper/product_data/EI.csv` and `paper/product_data/IE.csv` used in the generation of Figure 1B, run\
`python paper/get_product.py paper/rossi_data/EI.csv paper/znamenskiy_data/EI.csv -o paper/product_data/EI.csv`\
`python paper/get_product.py paper/rossi_data/EI.csv paper/znamenskiy_data/IE.csv -o paper/product_data/IE.csv`\
where the data file `paper/rossi_data/EI.csv` is generated by running\
`python paper/process_rossi.py paper/rossi_data -o paper/rossi_data`\
and the data files `paper/znamenskiy_data/EI.csv` and `paper/znamenskiy_data/IE.csv` are generated by running\
`python paper/process_znamenskiy.py paper/znamenskiy_data -o paper/znamenskiy_data`

To get the best-fit values and uncertainties of $\sigma_\mathrm{E} = 150 \pm 11$ and $\sigma_\mathrm{I} = 108 \pm 8$ as well as their geometric mean $\sqrt{\sigma_\mathrm{E}\sigma_\mathrm{I}} = 127.1 \pm 13.8$, run\
`python paper/fit_kernel.py paper/product_data/EI.csv paper/product_data/IE.csv`

To get the best-fit value of $\rho = 0.72$, run\
`python paper/uncertainty_rho.py 150.1898239023003 11.31920466 107.5719331775833 8.43638973`

To get the 95% confidenence interval $[0.443, 0.691]$ in Figure 2D and S1 (these values are used in the commands in producing those figures, as seen from the above table of commands), run\
`python paper/uncertainty.py paper/chettih_data/r0_samples.csv 127.1 13.8`\
where the file `paper/chettih_data/r0_samples.csv` is generated by running\
`python paper/estimator.py paper/chettih_data/data.csv -e r0 --rmax 100 -m 1 -o paper/chettih_data/r0_samples.csv`\
and the file `paper/chettih_data/data.csv` is generated by running\
`python paper/process_chettih.py paper/chettih_data/bias.csv --format csv -o paper/chettih_data/data.csv`\
Note that the values for the 95% confidence interval differs slightly from run to run since these values are estimated from bootstrapping and thus there is randomness involved.

Similarly, to get the 95% confidenence interval $[0.260, 0.530]$ in Figure 2E and S2 (these values are used in the commands in producing those figures, as seen from the above table of commands), run\
`python paper/uncertainty.py paper/chettih_data/r0_samples.csv paper/chettih_data/rmin_samples.csv 127.1 13.8 -e='f[1]-f[0]'`\
where the file `paper/chettih_data/rmin_samples.csv` is generated by running\
`python paper/estimator.py paper/chettih_data/data.csv -e rmin --rmax 300 -o paper/chettih_data/rmin_samples.csv`\
and the file `paper/chettih_data/data.csv` is generated by running\
`python paper/process_chettih.py paper/chettih_data/bias.csv --format csv -o paper/chettih_data/data.csv`\
Again, the values for the 95% confidence interval differs slightly from run to run since these values are estimated from bootstrapping and thus there is randomness involved.

To get the best-fit value and uncertainty of $\kappa_\mathrm{EE} = 0.198 \pm 0.054$ used in constraining fitted model parameters in Figure 6, run\
`python paper/fit_rossi_ori.py paper/rossi_data/fig2h.csv`
