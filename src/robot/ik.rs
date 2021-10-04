use std::f64::{INFINITY, NEG_INFINITY};
use std::future;

use hyperdual::Const;
use k::{self, Constraints, UnitQuaternion};
use optimization_engine::constraints::{Constraint, Rectangle};
use optimization_engine::{constraints::NoConstraints, panoc::*, *};
extern crate nalgebra as na;
use na::{SimdPartialOrd, Vector, Vector3};
use crate::robot::state::{State};
use crate::robot::constraints::{RobotConstraints, Vector7};
use super::constraints::{JointConstraint, ParsedConstraints};
use std::ops::Add;



type CostFunctionType = dyn Fn(&[f64], &mut f64) -> Result<(), SolverError>;
type Arm = k::SerialChain<f64>;


fn finite_difference(f: &dyn Fn(&[f64], &mut f64) -> Result<(), SolverError>, u: &[f64], grad: &mut [f64]) -> Result<(), SolverError> {
    let h = 1e-6;
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

    Ok(())
}


fn joint_constraint_cost(u: &[f64], jcs: &[JointConstraint; 7]) -> f64 {
    let mut c = 0.0;
    for (&v, jc) in u.iter().zip(jcs) {
        let center = jc.min.unwrap() + (jc.max.unwrap() - jc.min.unwrap()) / 2.;
        let w = jc.max.unwrap() - jc.min.unwrap();
        c += (2. * (v - center) / w).powf(50.);
    }
    c
}


fn finite_diff(lu: &[Option<f64>; 7], u: &[f64], dt: f64) -> [f64; 7] {
    let mut vel = [0.0; 7];

    for ((v, &olu), x) in vel.iter_mut().zip(lu).zip(u) {
        *v = (x - olu.unwrap()) / dt;
    }

    vel
}

fn constraint_cost(constriants : &Option<RobotConstraints>, state: &State, u: &[f64], dt: f64) -> f64 {
    let mut c = 0.0;

    if let Some(constraints) = &constriants {
        let vel = finite_diff(&state.joint_position, u, dt);
        let accel = finite_diff(&state.joint_velocity, &vel, dt);
        let jerk = finite_diff(&state.joint_acceleration, &accel, dt);

        // Position Constraints
        if let Some(jcs) = &constraints.joint_value {
            c += joint_constraint_cost(u, &jcs);
        }

        // Velocity Constraints
        if let Some(jcs) = &constraints.joint_vel {
            c += joint_constraint_cost(&vel, &jcs);
        }

        // Acceleration Constraints
        if let Some(jcs) = &constraints.joint_accel {
            c += joint_constraint_cost(&accel, &jcs);
        }

        // Jerk Constraints
        if let Some(jcs) = &constraints.joint_jerk {
            c += joint_constraint_cost(&jerk, &jcs);
        }
    }
    c
}


fn position_cost(current_position: &Vector3<f64>, desired_position: &Vector3<f64>) -> f64 {
    (current_position - desired_position).norm_squared()
}

fn rotation_cost(current_rotation: &UnitQuaternion<f64>, desired_rotation: &UnitQuaternion<f64>) -> f64 {
    current_rotation.angle_to(desired_rotation).powf(2.0)
}

fn goal_cost(desired_state: &State, arm: &Arm, dt: f64) -> f64 {
    let mut c = 0.0;
    if let Some(name_transforms) = &desired_state.transforms {
        for named_trans in name_transforms.iter() {
            let trans = arm.find(&named_trans.name).unwrap().world_transform().unwrap();

            if let Some(desired_position) = named_trans.position {
                c += position_cost(&trans.translation.vector, &desired_position);
            }
            
            if let Some(desired_rotation) = named_trans.rotation {
                c += rotation_cost(&trans.rotation, &desired_rotation);
            }
        }
    }
    c
}

fn div(a: &Vector7, b: &Vector7) -> Vector7 {
    a.zip_map(b, |_a, _b| _a / _b)
}

fn clip(a: &Vector7, lb: &Vector7, ub: &Vector7) -> Vector7 {
    a.sup(lb).inf(ub)
}

fn pow(a: &Vector7, p: f64) -> Vector7 {
    a.map(|_a| _a.powf(p))
}


fn max_j_v(v0: &Vector7, a0: &Vector7, v_max: &Vector7, j_max: &Vector7, dt: f64) -> Vector7 {
    let jm = -j_max;
    let A = -pow(&jm, -1.) * dt.powf(2.) / 2.;
    let B = -(pow(&jm, -1.).component_mul(&a0) * dt).add_scalar(-dt.powf(2.));
    let C = -(v_max - v0) + a0*dt - pow(&jm, -1.).component_mul(&pow(a0, 2.)) / 2.; 
    (-B + pow(&(pow(&B, 2.) - 4. * A.component_mul(&C)), 0.5)).component_div(&(2. * A))
}

fn max_tA_j(dq: &Vector7, v0: &Vector7, a0: &Vector7, a_max: &Vector7, j_max: &Vector7, dt: f64) -> Vector7 {
    let am = -a_max;
    let jm = -j_max;
    let tA = div(&am, &jm) - div(&clip(&(a0 ), &-a_max, &a_max), &jm);
    let Vc = v0 + a0.component_mul(&tA.add_scalar(dt)) + jm.component_mul(&tA.component_mul(&tA)) / 2.;

    let A = -div(&pow(&(dt * tA).add_scalar(dt.powf(2.)),2.), &(2. * am));
    let B = (pow(&tA, 2.) + tA * dt.powf(2.)).add_scalar(dt.powf(3.)) - Vc.component_mul(&(tA * dt).add_scalar(dt.powf(2.))).component_div(&am);
    let C = 
        -dq + 
        jm.component_mul(&pow(&tA, 3.)) / 6. +
        a0.component_mul(&((pow(&tA, 2.) / 2. + dt * tA).add_scalar(dt.powf(2.)))) -
        pow(&Vc, 2.).component_div(&(2. * am));
    
    (-B + pow(&(pow(&B, 2.) - 4. * A.component_mul(&C)), 0.5)).component_div(&(2. * A))
}

fn jerk_bound(q0: &Vector7, v0: &Vector7, a0: &Vector7, constraints: &ParsedConstraints, dt: f64) -> (Vector7, Vector7) {
    let zeros = Vector7::from([0.; 7]);
    let buffer = Vector7::from([1e-5; 7]);
    let dq = (constraints.joint_max - q0).sup(&zeros);

    let max_j = 
        ((constraints.joint_accel_max - a0 - buffer) / dt)
        .inf(&( (dq - v0 * dt - a0 * dt.powf(2.)) / dt.powf(3.) ))
        .inf(&max_j_v(v0, a0, &constraints.joint_vel_max, &constraints.joint_jerk_max, dt))
        .inf(&max_tA_j(&dq, v0, a0, &constraints.joint_accel_max, &constraints.joint_jerk_max, dt));



    let dq = (-constraints.joint_min + q0).sup(&zeros);
    let min_j = 
        ((constraints.joint_accel_max + a0 + buffer) / dt)
        .inf(&( (dq + v0 * dt + a0 * dt.powf(2.)) / dt.powf(3.) ))
        .inf(&max_j_v(&-v0, &-a0, &constraints.joint_vel_max, &constraints.joint_jerk_max, dt))
        .inf(&max_tA_j(&dq, &-v0, &-a0, &constraints.joint_accel_max, &constraints.joint_jerk_max, dt));


    (-min_j, max_j)
}

fn stopping_distance(v0: &Vector7, a0: &Vector7, constraints: &ParsedConstraints, dt: f64) -> Vector7 {
    let q = v0 * dt + a0 * dt.powf(2.) + constraints.joint_jerk_max * dt.powf(3.);
    let v = v0 + a0 * dt + constraints.joint_jerk_max * dt.powf(2.);
    let a = a0 + constraints.joint_jerk_max * dt;

    let jm = -constraints.joint_jerk_max;
    let am = -constraints.joint_accel_max;
    
    let tA = (am - a).component_div(&jm);
    let vA = v + a.component_mul(&tA) + jm.component_mul(&pow(&tA, 2.)) / 2.;

    let qA = q + v.component_mul(&tA) + a.component_mul(&pow(&tA, 2.)) / 2. + jm.component_mul(&pow(&tA, 3.)) / 6.;
    let tV = - vA.component_div(&am);
    let qTV = qA + vA.component_mul(&tV) + am.component_mul(&pow(&tV, 2.)) / 2.;

    let A = -27. * pow(&(jm / 6.), 2.);
    let B = 18. * (jm / 6.).component_mul(&(a / 2.)).component_mul(&v)
        - 4. * pow(&(a / 2.), 3.);
    let C = pow(&(a / 2.), 2.).component_mul(&pow(&v, 2.))
        - 4. * (jm / 6.).component_mul(&pow(&v, 3.));
    
    let desc = pow(&B, 2.) - 4. * A.component_mul(&C);


    let mut res = Vector7::from([0.; 7]);
    for i in 0..7 {
        if vA[i] > 0. {
            res[i] = qTV[i];
        } else {
            if desc[i] < 0. {
                res[i] = 0.;
            } else {
                res[i] = (-B[i] + (B[i].powf(2.) - 4. * A[i] * C[i]).sqrt()) / (2. * A[i]);
            }
        }
    }
    res
}

fn stopping_v(a0: &Vector7, constraints: &ParsedConstraints, dt: f64) -> Vector7 {
    let v = a0 * dt + constraints.joint_jerk_max * dt.powf(2.);
    let a = a0 + constraints.joint_jerk_max * dt;
    let jm = -constraints.joint_jerk_max;

    v - pow(&a, 2.).component_div(&jm) / 2.
}

fn compute_bounds(state: &State, constraints: &ParsedConstraints, dt: f64) -> ([f64;7], [f64;7]) {
    let q = Vector::from(state.joint_position.map(|p| p.unwrap()));
    let dq = Vector::from(state.joint_velocity.map(|p| p.unwrap()));
    let ddq = Vector::from(state.joint_acceleration.map(|p| p.unwrap()));

    let dt2 = dt * dt;
    let dt3 = dt2 * dt;

    let zeros = Vector7::from([0.; 7]);
    let s = 0.9;
    let buffer = Vector7::from([1e-4;7]);
    let suqb = constraints.joint_max - buffer - stopping_distance(&dq, &ddq, constraints, dt);
    let slqb = constraints.joint_min + buffer + stopping_distance(&-dq, &-ddq, constraints, dt);

    let suvb = constraints.joint_vel_max - buffer - stopping_v(&ddq, constraints, dt);
    let slvb = constraints.joint_vel_min + buffer + stopping_v(&-ddq, constraints, dt);
    
    let suab = constraints.joint_accel_max;
    let slab = constraints.joint_accel_min;

    let sujb = (suab - ddq).sup(&zeros);
    let sljb = (slab - ddq).inf(&zeros);

    let mut sub = suqb
        .inf(&(q + suvb * dt))
        .inf(&(q + dq * dt + suab * dt.powf(2.)))
        .inf(&(q + dq * dt + ddq * dt.powf(2.) + sujb * dt.powf(3.)));
    let mut slb = slqb
        .sup(&(q + slvb * dt))
        .sup(&(q + dq * dt + slab * dt.powf(2.)))
        .sup(&(q + dq * dt + ddq * dt.powf(2.) + sljb * dt.powf(3.)));

    let ujb = (&((constraints.joint_max - q - dq * dt - ddq * dt.powf(2.)) / dt.powf(3.)))
        .inf(&(((constraints.joint_vel_max - dq).sup(&zeros) - ddq * dt) / dt.powf(2.)))
        .inf(&(((constraints.joint_accel_max - ddq).sup(&zeros)) / dt))
        .inf(&constraints.joint_jerk_max);
    let ljb = (&((constraints.joint_min - q - dq * dt - ddq * dt.powf(2.)) / dt.powf(3.)))
        .sup(&(((constraints.joint_vel_min - dq).inf(&zeros) - ddq * dt) / dt.powf(2.)))
        .sup(&(((constraints.joint_accel_min - ddq).inf(&zeros)) / dt))
        .sup(&constraints.joint_jerk_min);


    let hub = q + dq * dt + ddq * dt2 + ujb * dt3;
    let hlb = q + dq * dt + ddq * dt2 + ljb * dt3;
    
    sub = sub.sup(&hlb);
    slb = slb.inf(&hub);

    for i in 0..7 {
        if sub[i] < slb[i] {
            let m = (sub[i] + slb[i]) / 2.;
            sub[i] = m;
            slb[i] = m;
        }
    }


    for i in 0..7 {
        if hub[i] < hlb[i] {
            println!("{:?}", q);
            println!("{:?}", constraints.joint_min);
            println!("{:?}", dq);
            println!("{:?}", constraints.joint_vel_max);
            println!("{:?}", ddq);

            println!("q: {:?}", q[i]);
            println!("ub: {:?}", hub[i]);
            println!("lb: {:?}", hlb[i]);

            println!("suvb: {:?}", suvb[i]);
            println!("suab: {:?}", suab[i]);

            println!("dq: {:?}", dq[i]);
            println!("ddq: {:?}", ddq[i]);
            panic!("");
        }
    }

    let mut lower_bound = [NEG_INFINITY; 7];
    let mut upper_bound = [INFINITY; 7];
    for i in 0..7 {
        lower_bound[i] = hlb[i].max(slb[i]);
        upper_bound[i] = hub[i].min(sub[i]);
    }


    (lower_bound, upper_bound)
}

pub struct Controller {
    panoc_cache : PANOCCache,
    arm : Arm,
    last_state : Option<State>,
    constriants : Option<ParsedConstraints>
}

impl Controller {
    pub fn new(arm: Arm, constraints : Option<ParsedConstraints>) -> Self {
        Controller {
            panoc_cache : panoc::PANOCCache::new(7, 1e-4, 100),
            arm: arm,
            last_state: None,
            constriants: constraints
        }
    }
    
    pub fn set_state(&mut self, state: State) {
        self.last_state = Some(state);
    }

    pub fn update(&mut self, state: &State, desired_state: &State, dt: f64) -> [f64; 7] {

        let arm = &mut self.arm;


        let mut ub = [INFINITY; 7];
        let mut lb = [NEG_INFINITY; 7];

        if let Some(constraints) = &self.constriants {
            for i in 0..7 {
                ub[i] = ub[i].min(constraints.joint_max[i]);
                lb[i] = lb[i].max(constraints.joint_min[i]);
            }
        }

        let cost = |u: &[f64], c: &mut f64| {

            // let mut clipped = [0.; 7];
            // for i in 0..7 {
            //     clipped[i] = u[i].max(lb[i]).min(ub[i]);
            // }

            if let Err(err) = arm.set_joint_positions(u) {
                println!("{}|{:?}", err, u);
                panic!();
            }
            arm.update_transforms();
            *c += goal_cost(desired_state, &arm, dt);
            Ok(())
        };

        let dcost = |u: &[f64], grad: &mut [f64]| {
            finite_difference(&cost, u, grad)
        };

        
        let mut u = [0.0; 7];
        for (ui, qi) in u.iter_mut().zip(state.joint_position) {
            *ui = qi.unwrap();
        }
        
        let status;

        if let Some(constraints) = &self.constriants {
            let (lb, ub) = compute_bounds(state, constraints, dt);
            let bounds = Rectangle::new(Some(&lb), Some(&ub));
            let problem = Problem::new(
                &bounds,
                 dcost, 
                cost
            );
            let mut panoc = PANOCOptimizer::new(problem, &mut self.panoc_cache)
                                .with_max_iter(10);
            status = panoc.solve(&mut u).unwrap();
        } else {
            let bounds = NoConstraints::new();
            let problem = Problem::new(
                &bounds,
                 dcost, 
                cost
            );
            let mut panoc = PANOCOptimizer::new(problem, &mut self.panoc_cache)
                                .with_max_iter(10);
            status = panoc.solve(&mut u).unwrap();
        }
        u
    }
}