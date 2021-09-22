use k;
use std::time::{Duration, Instant};
use optimization_engine::{constraints::*, panoc::*, *};
extern crate nalgebra as na;
use na::{Vector3, Rotation3, Rotation};

fn updateStuff(arm : &k::SerialChain<f64>) {
    let angles = vec![0.2, 0.2, 0.0, -1.0, 0.0, 0.0, 0.2];
    arm.set_joint_positions(&angles).unwrap();
    arm.update_transforms();
}


fn cost(u: &[f64], c: &mut f64) -> Result<(), SolverError> {
    
    Ok(())
}


fn finite_diff(f: &dyn Fn(&[f64], &mut f64) -> Result<(), SolverError>, u: &[f64], grad: &mut [f64]) {
    let h = 1e-4;
    let mut f0 = 0.0;
    f(u, &mut f0).unwrap();

    let mut x = [0.0,0.0,0.0,0.0,0.0,0.0,0.0];
    for i in 0..7 {
        x[i] = u[i];
    }

    for i in 0..7 {
        let mut fi = 0.0;
        x[i] += h;
        f(&x, &mut fi).unwrap();
        grad[i] = (fi - f0) / h;
        x[i] -= h;
    }
}

fn main() {
    let chain = k::Chain::<f64>::from_urdf_file("panda.urdf").unwrap();

    
    // Create a set of joints from end joint
    let ee = chain.find("panda_joint8").unwrap();
    let arm = k::SerialChain::from_end(ee);
    println!("chain: {}", arm);


    let mut goal = Vector3::new(0.5, 0.0, 0.5);

    let mut panoc_cache = PANOCCache::new(7, 1e-4, 100);

    let mut xmin = Vec::new();
    let mut xmax = Vec::new();
    for j in arm.iter_joints() {
        let range = (*j).limits.unwrap();
        xmin.push(range.min + 1e-2);
        xmax.push(range.max - 1e-2);
    }

    let bounds = constraints::Rectangle::new(Some(&xmin), Some(&xmax));
    let mut u = vec![0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0];
    
    arm.set_joint_positions(&u).unwrap();
    arm.update_transforms();

    let start = Instant::now();
    let N = 1000;
    for i in 0..N {
        goal[2] -= 0.001;
        let f = |u: &[f64], c: &mut f64| -> Result<(), SolverError> {
            arm.set_joint_positions(u).unwrap();
            arm.update_transforms();
            let trans = arm.end_transform();
            *c = (goal - trans.translation.vector).norm_squared();
            Ok(())
        };


        let df = |u: &[f64], grad: &mut [f64]| -> Result<(), SolverError> {
            finite_diff(&f, u, grad);
    
            Ok(())
        };
    

        let problem = Problem::new(&bounds, df, f);
        let mut panoc = PANOCOptimizer::new(problem, &mut panoc_cache)
                            .with_max_iter(1);
        let status = panoc.solve(&mut u).unwrap();
        arm.set_joint_positions(&u).unwrap();
    }
    let duration = start.elapsed();
    println!("{}", 1e6 * N as f64 / (duration.as_micros() as f64));
}
