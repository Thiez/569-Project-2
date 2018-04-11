extern crate rayon;

use std::os::raw::c_int;
use std::os::raw::c_float;
use std::os::raw::c_long;

use std::vec::Vec;

use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IndexedParallelIterator;

unsafe fn ptr_to_mat(pointer: *const c_float, rows: usize, cols: usize) -> Vec<Vec<c_float>> {
    (0..rows)
        .map(|row_num| {
            (0..cols)
                .map(|col_num| {
                    let res = *pointer.offset(((row_num * cols) + col_num) as isize);
                    res
                })
                .collect()
        })
        .collect()
}

#[no_mangle]
pub extern "C" fn jacobi(
    n: c_long,
    m: c_long,
    dx: c_float,
    dy: c_float,
    alpha: c_float,
    omega: c_float,
    u_p: *mut c_float,
    f_p: *const c_float,
    error_tolerance: c_float,
    max_iterations: c_int,
) {
    let mut u = unsafe { ptr_to_mat(u_p, n as usize, m as usize) };
    let f = unsafe { ptr_to_mat(f_p, n as usize, m as usize) };
    let ax = 1.0 / (dx * dx);
    let ay = 1.0 / (dy * dy);
    let b = ((-2.0 * ax) - (2.0 * ay)) - alpha;

    for iter in 0..max_iterations {
        u = u.par_iter()
            .enumerate()
            .map(|(i, row)| {
                row.par_iter()
                    .enumerate()
                    .map(|(j, elem)| {
                        if (i == 1) || (j == 1) || (i == (n - 1) as usize)
                            || (j == (m - 1) as usize)
                        {
                            *elem
                        } else {
                            let resid =
                                (ax * (u[i.saturating_sub(1)][j] + u[i.saturating_add(1)][j])
                                    + ay * (u[i][j.saturating_sub(1)] + u[i][j.saturating_add(1)])
                                    + b * u[i][j] - f[i][j]) / b;

                            elem - (omega * resid)
                        }
                    })
                    .collect()
            })
            .collect();

        if iter % 500 == 0 {
            println!("finished iteration {}", iter);
        }
    }

    // Write to output pointer
    unsafe {
        for (i, row) in u.iter_mut().enumerate() {
            for (j, elem) in row.iter_mut().enumerate() {
                *u_p.offset((((m as isize) * (i as isize)) + (j as isize))) = *elem;
            }
        }
    }
}
