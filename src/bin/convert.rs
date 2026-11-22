//! Convert a safetensors checkpoint to a packed ternary `.tgm` file.
//!
//! usage: convert <in.safetensors> <out.tgm> [--per-row] [--keep-other-2d] [--patterns a,b,c]

use ternary_gemm::convert::{convert_safetensors, ConvertOptions};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!("usage: convert <in.safetensors> <out.tgm> [--per-row] [--keep-other-2d] [--patterns a,b,c]");
        std::process::exit(2);
    }
    let mut o = ConvertOptions::default();
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--per-row" => o.per_row = true,
            "--keep-other-2d" => o.keep_other_2d = true,
            "--patterns" => {
                i += 1;
                o.ternary_patterns = args
                    .get(i)
                    .map(|s| s.split(',').map(String::from).collect())
                    .unwrap_or_default();
            }
            x => {
                eprintln!("unknown argument {x}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let bytes = std::fs::read(&args[0]).unwrap_or_else(|e| {
        eprintln!("read {}: {e}", args[0]);
        std::process::exit(1)
    });
    let (tensors, rep) = convert_safetensors(&bytes, &o).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1)
    });
    ternary_gemm::format::save(&args[1], &tensors).unwrap_or_else(|e| {
        eprintln!("write {}: {e}", args[1]);
        std::process::exit(1)
    });
    for (n, rows, k, sp) in &rep.ternarized {
        println!("ternary  {n:<60} {rows}x{k}  zeros {:.1}%", sp * 100.0);
    }
    for n in &rep.kept_f32 {
        println!("f32      {n}");
    }
    for (n, why) in &rep.skipped {
        println!("skipped  {n}: {why}");
    }
    println!(
        "{} ternary, {} f32, {} skipped; {:.1} MB in -> {:.1} MB out",
        rep.ternarized.len(),
        rep.kept_f32.len(),
        rep.skipped.len(),
        rep.input_bytes as f64 / 1e6,
        rep.packed_bytes as f64 / 1e6
    );
}
