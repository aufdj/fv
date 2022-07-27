use std::{
    path::{Path, PathBuf},
    time::Instant,
    io::{
        Read, Write, BufRead,
        BufWriter, BufReader, 
        ErrorKind,
    },
    fs::File,
    process::exit,
};


/// A trait for handling buffered reading.
pub trait BufferedRead {
    fn read_byte(&mut self) -> u8;
}
impl BufferedRead for BufReader<File> {
    /// Read one byte from an input file.
    fn read_byte(&mut self) -> u8 {
        let mut byte = [0u8; 1];

        if self.read(&mut byte).is_ok() {
            if self.buffer().is_empty() {
                self.consume(self.capacity());

                if let Err(e) = self.fill_buf() {
                    println!("Error: {}", e);
                }
            }
        }
        else {
            println!("Function read_byte failed.");
        }
        u8::from_le_bytes(byte)
    }
}

/// A trait for handling buffered writing.
pub trait BufferedWrite {
    fn write_byte(&mut self, output: u8);
    fn write_u16(&mut self, output: u16);
    fn write_u32(&mut self, output: u32);
    fn flush_buffer(&mut self);
}
impl BufferedWrite for BufWriter<File> {
    /// Write one byte to an output file.
    fn write_byte(&mut self, output: u8) {
        if let Err(e) = self.write(&[output]) {
            println!("Error: {}", e);
        }
        
        if self.buffer().len() >= self.capacity() {
            if let Err(e) = self.flush() {
                println!("Error: {}", e);
            }
        }
    }
    fn write_u16(&mut self, output: u16) {
        if let Err(e) = self.write(&output.to_le_bytes()[..]) {
            println!("Error: {}", e);
        }
        
        if self.buffer().len() >= self.capacity() {
            if let Err(e) = self.flush() {
                println!("Error: {}", e);
            }
        }
    }
    fn write_u32(&mut self, output: u32) {
        if let Err(e) = self.write(&output.to_le_bytes()[..]) {
            println!("Error: {}", e);
        }
        
        if self.buffer().len() >= self.capacity() {
            if let Err(e) = self.flush() {
                println!("Error: {}", e);
            }
        }
    }

    /// Flush buffer to file.
    fn flush_buffer(&mut self) {
        if let Err(e) = self.flush() {
            println!("Error: {}", e);
        }    
    }
}
/// Takes a file path and returns an input file wrapped in a BufReader.
pub fn new_input_file(file_path: &Path) -> BufReader<File> {
    BufReader::with_capacity(
        4096, 
        match File::open(file_path) {
            Ok(file) => file,
            Err(e) => match e.kind() {
                ErrorKind::NotFound => {
                    println!("Couldn't open file {}: Not Found", file_path.display());
                    exit(0); 
                }
                ErrorKind::PermissionDenied => {
                    println!("Couldn't open file {}: Permission Denied", file_path.display());
                    exit(0);
                }
                _ => {
                    println!("Couldn't open file {}", file_path.display());
                    exit(0);
                }
            }
        }
    )
}
/// Takes a file path and returns an output file wrapped in a BufWriter.
pub fn new_output_file(file_path: &Path) -> BufWriter<File> {
    BufWriter::with_capacity(
        4096, 
        match File::create(file_path) {
            Ok(file) => file,
            Err(e) => match e.kind() {
                ErrorKind::NotFound => {
                    println!("Couldn't open file {}: Not Found", file_path.display());
                    exit(0); 
                }
                ErrorKind::PermissionDenied => {
                    println!("Couldn't open file {}: Permission Denied", file_path.display());
                    exit(0);
                }
                _ => {
                    println!("Couldn't open file {}", file_path.display());
                    exit(0);
                }  
            }
        }
    )
}



enum Parse {
    None,
    Inputs,
    ColScale,
    Width,
}

#[derive(Clone, Debug)]
pub struct Fv {
    pub col_scale:  f64,
    pub width:      i32,
    pub inputs:     Vec<PathBuf>,
}
impl Default for Fv {
    fn default() -> Fv {
        Fv {
            col_scale:  10.0,
            width:      512,
            inputs:     Vec::new(),
        }
    }
}


fn parse(args: Vec<String>) -> Fv {
    let mut fv = Fv::default();
    let mut parser = Parse::None;

    for arg in args.into_iter() {
        match arg.as_str() {
            "-col-scale" => {
                parser = Parse::ColScale;
                continue;
            }
            "-width" => {
                parser = Parse::Width;
                continue;
            }
            "-i" | "-inputs" => { 
                parser = Parse::Inputs;
                continue;
            },
            _ => {},
        }
        match parser {
            Parse::ColScale => {
                if let Ok(c) = arg.parse::<f64>() {
                    fv.col_scale = c;
                }
                else {
                    println!("{arg} is not a valid color scale.");
                    exit(0);
                }
            }
            Parse::Width => {
                if let Ok(w) = arg.parse::<i32>() {
                    fv.width = w;
                }
                else {
                    println!("{arg} is not a valid width.");
                    exit(0);
                }
            }
            Parse::Inputs => {
                let path = PathBuf::from(&arg);
                if path.exists() {
                    fv.inputs.push(path);
                }
                else {
                    println!("{arg} is not a valid input.");
                    exit(0);
                }
            }
            Parse::None => {},
        }
    }

    if fv.inputs.is_empty() {
        println!("No inputs found.");
        exit(0);
    }

    while expand(&mut fv.inputs).is_some() {}

    fv
}


/// Removes all directories from inputs and pushes the contents of the
/// directories to inputs. An additional iteration is needed for each 
/// level of nested directories.
fn expand(inputs: &mut Vec<PathBuf>) -> Option<usize> {
    let mut dirs: Vec<(usize, PathBuf)> = Vec::new();

    for (i, input) in inputs.iter().enumerate() {
        if input.is_dir() {
            dirs.push((i, input.clone()));
        }
    }
    for (i, dir) in dirs.iter() {
        for file in dir.read_dir().unwrap().map(|d| d.unwrap().path()) {
            inputs.push(file);
        }
        inputs.swap_remove(*i);
    }
    if !dirs.is_empty() {
        return Some(0);
    }
    None
}


fn main() {
    let args = std::env::args().skip(1).collect::<Vec<String>>();
    let fv = parse(args);

    println!();
    println!("Visualizing:");
    for input in fv.inputs.iter() {
        println!("  {}", input.display());
    }
    println!();

    new(fv);
}


/* fv.cpp - File entropy visualization program

Copyright (C) 2006, 2014, Matt Mahoney.  This program is distributed
without warranty under terms of the GNU general public license v3.
See http://www.gnu.org/licenses/gpl.txt

Usage: fv file (Requires 512 MB memory)

The output is a .bmp with the given size in pixels, which visually
displays where matching substrings of various lengths and offests are
found.  A pixel at x, y is (black, red, green, blue) if the last matching
substring of length (1, 2, 4, 8) at x occurred y bytes ago.  x and y
are scaled so that the image dimensions match the file length.
The y axis is scaled log base 10.  The maximum range of a match is 1 GB.
*/

// Hash table size, needs HSIZE*4 bytes of memory (512 MB).
// To reduce memory usage, use smaller power of 2. This may cause the 
// program to miss long range matches in very large files, but won't 
// affect smaller files.
const HSIZE: usize = 0x8000000;


/// Linear Congruential Generator
// https://www.rosettacode.org/wiki/Linear_congruential_generator#Rust
struct Rand {
    state: u32,
}
impl Rand {
    fn seed(state: u32) -> Rand {
        Rand { state }
    }
    // Generate random number in 0..32767
    fn next(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(214013).wrapping_add(2531011);
        self.state %= 1 << 31;
        (self.state >> 16) as f64
    }
}


// ilog(x) = (x.ln()*c) in range 0 to 255, faster than direct computation
struct Ilog {
    t: [f64; 256]
}
impl Ilog {
    fn new(c: f64) -> Ilog {
        let mut t = [0.0; 256];
        for i in 0..256 {
            t[i] = (i as f64 / c).exp();
        }
        Ilog { t }
    }
    fn ilog(&self, x: f64) -> i32 {
        // Find i such that t[i-1] < x <= t[i] by binary search
        let mut i = 128;  
        if x < self.t[i] { i-=128; }
        i+=64;
        if x < self.t[i] { i-=64; }
        i+=32;
        if x < self.t[i] { i-=32; }
        i+=16;
        if x < self.t[i] { i-=16; }
        i+=8;
        if x < self.t[i] { i-=8; }
        i+=4; 
        if x < self.t[i] { i-=4; }
        i+=2;
        if x < self.t[i] { i-=2; }
        i+=1;
        if x < self.t[i] { i-=1; }
        i as i32
    }
}

fn clamp(c: i32) -> u8 {
    if c > 255 { 
        255 
    } 
    else if c < 0 { 
        0 
    }
    else { 
        c as u8 
    }
}

struct Image {
    pixels:  Vec<u8>,
    width:   i32,
    height:  i32,
}
impl Image {
    fn new(w: i32, h: i32) -> Image {
        Image {
            pixels:  vec![0u8; (w*h*3) as usize],
            width:   w,
            height:  h,
        }
    }
    fn put(&mut self, x: i32, y: i32, red: i32, green: i32, blue: i32) {
        assert!(x >= 0);
        assert!(y >= 0);
        assert!(x < self.width);
        assert!(y < self.height);

        let row = y * self.width;
        let i = ((row + x) * 3) as usize;

        let mut c = self.pixels[i] as i32 + blue;
        self.pixels[i] = clamp(c);

        c = self.pixels[i+1] as i32 + green;
        self.pixels[i+1] = clamp(c);

        c = self.pixels[i+2] as i32 + red;
        self.pixels[i+2] = clamp(c);
    }
    fn save_bmp(&mut self, file_out: &mut BufWriter<File>) {
        let file_size  = (54 + self.pixels.len()) as u32;
        let image_size = (self.width * self.height * 3) as u32;

        file_out.write_u16(u16::from_le_bytes(*b"BM"));
        file_out.write_u32(file_size);          // File size
        file_out.write_u32(0);                  // Reserved
        file_out.write_u32(54);                 // Offset to start of image (no palette)
        file_out.write_u32(40);                 // Info header size
        file_out.write_u32(self.width as u32);  // Image width in pixels
        file_out.write_u32(self.height as u32); // Image height in pixels
        file_out.write_u16(1);                  // Image planes
        file_out.write_u16(24);                 // Output bits per pixel
        file_out.write_u32(0);                  // No compression
        file_out.write_u32(image_size);         // Image size in bytes
        file_out.write_u32(3000);               // X pixels per meter
        file_out.write_u32(3000);               // Y pixels per meter
        file_out.write_u32(0);                  // Colors
        file_out.write_u32(0);                  // Important colors
        for pixel in self.pixels.iter() {
            file_out.write_byte(*pixel);
        }
    }
}

/// Return the length of a file.
pub fn file_len(path: &Path) -> u64 {
    path.metadata().unwrap().len()
}

pub fn new(fv: Fv) {
    let name = fv.inputs[0]
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap();
    let file_name = &format!("{name}.bmp");
    let mut file_out = new_output_file(&Path::new(file_name));
    let size_sum  = fv.inputs.iter().map(|f| file_len(f)).sum();
    let fsize_sum = size_sum as f64;

    let width   = fv.width;
    let height  = 256i32;
    let fwidth  = width  as f64;
    let fheight = height as f64;

    println!("Drawing {} by {} image {}",
        width, 
        height,
        file_name
    );
    let time = Instant::now();
    
    // Create blank white image
    let mut img = Image::new(width, height);
    for i in 0..width {
        for j in 0..height {
            img.put(i, j, 255, 255, 255);
        }
    }

    // Draw tick marks on the Y axis (log base 10 scale)
    let y_label_width: i32 = width / 50; // Tick mark width
    let mut i = 1;
    if height >= 2 && width >= 2 {
        loop {
            for j in 1i32..10 {
                let log_pos = i * j as u64; // 1, 2, 3.., 10, 20, 30.., 100, 200, 300..
                if log_pos < size_sum {
                    let a = (log_pos as f64).ln();
                    let b = fsize_sum.ln();
                    let y = (fheight * a / b) as i32;
                    // Darken a horizontal line of pixels to create a tick mark.
                    // Marks start as black and progressively turn lighter gray,
                    // returning to black at each new order of magnitude.
                    for x in 0..y_label_width {
                        img.put(x, y, -255/j, -255/j, -255/j);
                    }    
                }
            }
            if i * 10 > size_sum { break; }
            i *= 10;
        }
    }
    
    // Darken x,y where there are matching strings at x and y (scaled) in s
    let csd      = fv.col_scale * fwidth * fheight / (fsize_sum + 0.5); // Color scale
    let cs       = csd as i32 + 1; // Rounded color scale
    let l2       = fheight / (2.0 + fsize_sum).ln();
    let ilog     = Ilog::new(l2);
    let xscale   = fwidth * 0.98 / fsize_sum; // Scale x axis so file fits within image width
    let mut ht   = vec![0u32; HSIZE]; // Hash -> checksum (high 2 bits), location (low 30 bits)
    let csd_max  = csd * 32767.0;
    let rec_max  = 1.0 / 32767.0;
    let mut rand = Rand::seed(1);

    let mut h: u32 = 0; // Hash
    let mut sum = 0;

    // Do 4 passes through file, first finding 1 byte matches (black),
    // then 2 (red), then 4 (green), then 8 (blue).
    for i in 0i32..4 {
        let start_pass = Instant::now();
        
        let mut xd: f64 = y_label_width as f64 - xscale; // Starting x position
        if i >= 2 {
            for i in ht.iter_mut() { 
                *i = 0; 
            }
        }
        for file in fv.inputs.iter() {
            let mut file_in = new_input_file(&file);
            sum += file_len(file);
            
            for j in sum..sum+file_len(file) {
                let c = file_in.read_byte() as u32;
                h = match i {
                    0 => c + 0x10000,                  // Hash of last byte,    1st pass
                    1 => (h * 256 + c) & 0xFFFF,       // Hash of last 2 bytes, 2nd pass
                    2 => h * 29 * 256 + c + 1,         // Hash of last 4 bytes, 3rd pass
                    _ => h * (16 * 123456789) + c + 1, // Hash of last 8 bytes, 4th pass
                };

                xd += xscale; // Move to next x position

                let prev_loc = &mut ht[((h ^ (h >> 16)) as usize) & (HSIZE-1)];
                let chksum   = (h & 0xC0000000) as u32;
                let new_loc  = chksum as u64 + j;

                if *prev_loc > chksum && *prev_loc < new_loc as u32 {
                    let x = xd as i32;
                    let r = rand.next() * rec_max; // 0..1
                    let y = ilog.ilog(r + (new_loc as i64 - *prev_loc as i64) as f64);

                    if cs > 1 || rand.next() < csd_max {
                        match i {
                            0 => img.put(x, y, -cs, -cs, -cs), // Black, 1st pass 
                            1 => img.put(x, y, 0,   -cs, -cs), // Red,   2nd pass
                            2 => img.put(x, y, -cs, 0,   -cs), // Green, 3rd pass 
                            _ => img.put(x, y, -cs, -cs, 0  ), // Blue,  4th pass
                        }
                    }  
                }
                *prev_loc = new_loc as u32;
            }
        } 
        println!("Drew part {} of 4 in {:.2?}",
            i + 1, 
            start_pass.elapsed()
        );
    }
    img.save_bmp(&mut file_out);
    
    println!("Created {} in {:.2?}", 
        file_name,
        time.elapsed()
    );
}
