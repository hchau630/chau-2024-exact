Commands for reproducing paper figures

| Figure | Command |
|--------|---------|
| 1B | `python paper/fit_kernel.py paper/product_data/EI.csv paper/product_data/IE.csv -o paper/figures/1b.pdf` |
| 1D (left) | `python paper/lineplot.py compare -d 2 -s 125 90 85 110 --wee 3 --wei 4 --wie 4 --wii 5.25 --kee 0.5 --kei -0.25 --kie -0.25 --kii 0.25 --N-ori 12 --N-osi 7 --approx-order 3 -o paper/figures/1d_a.pdf` |
| 1D (right) | `python paper/lineplot.py compare -d 2 -s 125 90 85 110 --wee 3 --wei 4 --wie 4 --wii 5.25 --kee 0.5 --kei -0.25 --kie -0.25 --kii 0.25 --N-ori 12 --N-osi 7 --approx-order 3 -m ori_osi -o paper/figures/1d_b.pdf` |
| 2A | `python paper/contourplot.py space_2 --rho 1 -o paper/figures/2a.pdf` |
| 2B | `python paper/contourplot.py space_2 --rho 1 2 -o paper/figures/2b.pdf` |
| 2C | `python paper/lineplot.py response --wee 4.5 1.5 1.5 --wei 3 3 2 --wie 2 2.5 2.5 --wii 0 4 4 -s 40 40 --axsize 2.25 1.85 -o paper/figures/2c.pdf` |
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
| 5D | Same as Figure 4B