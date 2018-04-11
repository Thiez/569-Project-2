extern crate libc;

use std::ops::{Index, IndexMut};
use libc::{c_int, c_float, c_long, free, malloc, memcpy, printf};

struct Matrix<T> {
    p: T,
    rows: usize,
}

impl<T> Index<(usize, usize)> for Matrix<*const T> {
    type Output = T;
    fn index(&self, (n, m): (usize, usize)) -> &Self::Output {
        unsafe {
            &*self.p.offset((n*self.rows+m) as isize)
        }
    }
}

impl<T> Index<(usize, usize)> for Matrix<*mut T> {
    type Output = T;
    fn index(&self, (n, m): (usize, usize)) -> &Self::Output {
        unsafe {
            &*self.p.offset((n*self.rows+m) as isize)
        }
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<*mut T> {
    fn index_mut(&mut self, (n, m): (usize, usize)) -> &mut <Self as Index<(usize, usize)>>::Output {
        unsafe {
            &mut *self.p.offset((n*self.rows+m) as isize)
        }
    }
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
    let n = n as usize;
    let m = m as usize;
    let mut u = Matrix { p: u_p, rows: n };
    let f = Matrix { p: f_p, rows: n };
    let ax = 1.0 / (dx * dx);
    let ay = 1.0 / (dy * dy);
    let b = ((-2.0 * ax) - (2.0 * ay)) - alpha;

    let u_size = n * m * std::mem::size_of::<c_float>();
    let u_old = unsafe {
        malloc(u_size)
    };
    for iter in 0..max_iterations {
        unsafe {
            memcpy(u_old, u.p as *const _, u_size);
        }
        let u_old = Matrix { p: u_old as *const c_float, rows: n };
        for i in 1..(n-1) {
            for j in 1..(m-1) {
                let resid = (ax * (u_old[(i-1, j)] + u_old[(i+1, j)])
                        + ay * (u_old[(i, j -1)] + u_old[(i, j+1)])
                        + b * u_old[(i,j)] - f[(i,j)]) / b;
                u[(i, j)] = u_old[(i,j)] - omega * resid;
            }
        }

        if iter % 500 == 0 {
            unsafe { printf(&"finished iteration %d\n\0".as_bytes()[0] as *const _ as *const _, iter as c_int); }
        }
    }
    unsafe {
        free(u_old);
    }
}