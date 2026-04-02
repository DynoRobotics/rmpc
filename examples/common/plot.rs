use std::fmt::Display;
use std::fs::File;
use std::io::{Result, Write};
use std::path::Path;
use std::process::{Command, Stdio};

use rmpc::{ArrayInst, FieldNames};

/// A helper to plot multiple signals using Gnuplot.
///
/// This is only meant for prototyping and debugging purposes. The API is
/// intentionally kept small and inflexible. If you want nice looking plots,
/// use a proper plotting tool instead.
pub struct Plot {
    dt: f64,
    series: Vec<Series>,
}

/// The type of plot to draw.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PlotType {
    /// A continuous line.
    Lines,
    /// A staircase.
    Stairs,
}

struct Series {
    kind: PlotType,
    label: String,
    values: Vec<f64>,
}

impl Plot {
    /// Creates a new instance of `Plot` with the specified time delta.
    pub fn with_dt(dt: f64) -> Self {
        Plot {
            dt,
            series: Vec::new(),
        }
    }

    /// Adds a new signal
    pub fn values(mut self, kind: PlotType, label: impl Display, data: Vec<f64>) -> Self {
        self.series.push(Series {
            kind,
            label: label.to_string(),
            values: data,
        });
        self
    }

    /// Adds all the signals in the struct using the field names as labels.
    #[expect(unused)]
    pub fn structs<A: ArrayInst<Item = f64> + FieldNames>(
        mut self,
        kind: PlotType,
        states: Vec<A>,
    ) -> Self {
        for (i, name) in A::FIELD_NAMES.iter().enumerate() {
            self = self.values(kind, name, states.iter().map(|s| s.as_slice()[i]).collect());
        }
        self
    }

    /// Creates a plot of the signals using Gnuplot, saving them as an SVG.
    pub fn plot_svg(self, path: impl AsRef<Path>) -> Result<()> {
        let mut command = Command::new("gnuplot")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;

        let mut stdin = command.stdin.take().unwrap();

        let num_plots = self.series.len();
        writeln!(
            stdin,
            "set terminal svg background rgb 'white' size 700, {height}",
            height = 200 * num_plots,
        )?;
        writeln!(stdin, "set multiplot layout {num_plots},1")?;

        writeln!(stdin, "set tmargin 1")?;
        writeln!(stdin, "set bmargin 2")?;
        writeln!(stdin, "set lmargin 12")?;
        writeln!(stdin, "set rmargin 3")?;
        writeln!(stdin, "set offsets graph 0, 0, 0.1, 0.1")?;

        let mut max_t = 0_f64;
        for series in &self.series {
            let last_step = match series.kind {
                PlotType::Lines => series.values.len(),
                PlotType::Stairs => series.values.len() + 1,
            };

            max_t = max_t.max((last_step - 1) as f64 * self.dt);
        }

        writeln!(stdin, "set xr[0:{max_t}]")?;

        for series in self.series {
            let min = *series
                .values
                .iter()
                .filter(|v| v.is_finite())
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(&0.0);
            let max = *series
                .values
                .iter()
                .filter(|v| v.is_finite())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(&0.0);

            let margin = if min != max { 0.1 * (max - min) } else { 0.001 };

            writeln!(stdin, "set yr[{}:{}]", min - margin, max + margin)?;

            writeln!(stdin, "$DATA << END")?;

            for (i, data) in series.values.into_iter().enumerate() {
                match series.kind {
                    PlotType::Lines => writeln!(stdin, "{} {}", i as f64 * self.dt, data)?,
                    PlotType::Stairs => {
                        writeln!(stdin, "{} {}", i as f64 * self.dt, data)?;
                        writeln!(stdin, "{} {}", (i + 1) as f64 * self.dt, data)?;
                    }
                };
            }
            writeln!(stdin, "END")?;

            writeln!(stdin, "set ylabel {:?}", series.label)?;
            writeln!(stdin, "plot $DATA with lines notitle linecolor rgb 'navy'")?;
        }

        writeln!(stdin, "unset multiplot")?;

        drop(stdin);

        let path = path.as_ref();

        let mut stdout = command.stdout.take().unwrap();
        std::io::copy(&mut stdout, &mut File::create(path)?)?;

        println!("Plot saved to {}", path.display());

        Ok(())
    }
}
