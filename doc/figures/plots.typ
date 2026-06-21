//! Document for rendering the plots in the usage guide.
//!
//! To render the plots, navigate to this directory and run the command
//! ```
//! typst compile --format svg plots.typ figure-{p}.svg
//! ```

#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 0.5em)
#show: lq.set-diagram(width: 7cm, height: 5cm)

#let mpc-plot(name, title) = {
  let data = csv(name + ".csv", row-type: dictionary)
    .map(row => row.map(float));

  let x = lq.linspace(0, 10)
  let y = x.map(x => calc.sin(0.1 * x * x))

  lq.diagram(
    title: title,
    aspect-ratio: 1.0,
    legend: (position: top + left),
    xaxis: (tick-distance: 0.5),
    yaxis: (tick-distance: 0.5),

    lq.plot(
      color: lq.color.map.petroff10.at(3),
      data.map(r => r.target_x),
      data.map(r => r.target_y),
      label: [Target],
    ),
    lq.plot(
      color: lq.color.map.petroff10.at(0),
      data.map(r => r.opt_x),
      data.map(r => r.opt_y),
      label: [Solution],
    ),
    lq.plot(
      color: lq.color.map.petroff10.at(1),
      data.map(r => r.sim_x),
      data.map(r => r.sim_y),
      label: [Simulated],
    ),
  )
}

#page(mpc-plot("mpc-linear", [Linear MPC]))
#page(mpc-plot("mpc-penalty", [Penalty Function]))
#page(mpc-plot("mpc-filter", [Filter Method]))
#page(mpc-plot("mpc-trust", [Trust Region]))
