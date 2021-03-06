use std::{
    path::Path,
    time::Instant,
    io::{
        Read, Write, BufRead,
        BufWriter, BufReader, 
        ErrorKind,
        Seek, SeekFrom,
    },
    fs::File,
    process::exit,
};


/* fv.cpp - File entropy visualization program

Copyright (C) 2006, 2014, Matt Mahoney.  This program is distributed
without warranty under terms of the GNU general public license v3.
See http://www.gnu.org/licenses/gpl.txt

Usage: fv file (Requires 512 MB memory)

The output is fv.bmp with the given size in pixels, which visually
displays where matching substrings of various lengths and offests are
found.  A pixel at x, y is (black, red, green, blue) if the last matching
substring of length (1, 2, 4, 8) at x occurred y bytes ago.  x and y
are scaled so that the image dimensions match the file length.
The y axis is scaled log base 10.  The maximum range of a match is 1 GB.
*/


/// A trait for handling buffered reading.
pub trait BufferedRead {
    fn read_byte(&mut self) -> u8;
}
impl BufferedRead for BufReader<File> {
    /// Read one byte from an input file.
    fn read_byte(&mut self) -> u8 {
        let mut byte = [0u8; 1];
        match self.read(&mut byte) {
            Ok(_)  => {},
            Err(e) => {
                println!("Function read_byte failed.");
                println!("Error: {}", e);
            },
        };
        if self.buffer().is_empty() {
            self.consume(self.capacity());
            match self.fill_buf() {
                Ok(_)  => {},
                Err(e) => {
                    println!("Function read_byte failed.");
                    println!("Error: {}", e);
                },
            }
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
        match self.write(&[output]) {
            Ok(_)  => {},
            Err(e) => {
                println!("Function write_byte failed.");
                println!("Error: {}", e);
            },
        }
        if self.buffer().len() >= self.capacity() {
            match self.flush() {
                Ok(_)  => {},
                Err(e) => {
                    println!("Function write_byte failed.");
                    println!("Error: {}", e);
                },
            }
        }
    }
    fn write_u16(&mut self, output: u16) {
        match self.write(&output.to_le_bytes()[..]) {
            Ok(_)  => {},
            Err(e) => {
                println!("Function write_byte failed.");
                println!("Error: {}", e);
            },
        }
        if self.buffer().len() >= self.capacity() {
            match self.flush() {
                Ok(_)  => {},
                Err(e) => {
                    println!("Function write_byte failed.");
                    println!("Error: {}", e);
                },
            }
        }
    }
    fn write_u32(&mut self, output: u32) {
        match self.write(&output.to_le_bytes()[..]) {
            Ok(_)  => {},
            Err(e) => {
                println!("Function write_byte failed.");
                println!("Error: {}", e);
            },
        }
        if self.buffer().len() >= self.capacity() {
            match self.flush() {
                Ok(_)  => {},
                Err(e) => {
                    println!("Function write_byte failed.");
                    println!("Error: {}", e);
                },
            }
        }
    }

    /// Flush buffer to file.
    fn flush_buffer(&mut self) {
        match self.flush() {
            Ok(_)  => {},
            Err(e) => {
                println!("Function flush_buffer failed.");
                println!("Error: {}", e);
            },
        }    
    }
}
/// Takes a file path and returns an input file wrapped in a BufReader.
pub fn new_input_file(capacity: usize, file_path: &Path) -> BufReader<File> {
    BufReader::with_capacity(
        capacity, 
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
pub fn new_output_file(capacity: usize, file_path: &Path) -> BufWriter<File> {
    BufWriter::with_capacity(
        capacity, 
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
/// Return the length of a file.
pub fn file_len(path: &Path) -> u64 {
    path.metadata().unwrap().len()
}

// Hash table size, needs HSIZE*4 bytes of memory (512 MB).
// To reduce memory usage, use smaller power of 2. This may cause the 
// program to miss long range matches in very large files, but won't 
// affect smaller files.
const HSIZE: usize = 0x8000000;  


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

        let i = ((y * self.width + x) * 3) as usize;

        let mut c = self.pixels[i] as i32 + blue;
        self.pixels[i] = 
        if c > 255 { 255 } 
        else if c < 0 { 0 }
        else { c as u8 };

        c = self.pixels[i+1] as i32 + green;
        self.pixels[i+1] =
        if c > 255 { 255 } 
        else if c < 0 { 0 }
        else { c as u8 };

        c = self.pixels[i+2] as i32 + red;
        self.pixels[i+2] =
        if c > 255 { 255 } 
        else if c < 0 { 0 }
        else { c as u8 };
    }
    fn save_bmp(&mut self, file_name: &str) {
        let mut file_out = new_output_file(4096, Path::new(file_name));
        let file_size    = (54 + self.pixels.len()) as u32;
        let image_size   = (self.width * self.height * 3) as u32;

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

fn main() {
    let time        = Instant::now();
    let args        = std::env::args().skip(1).collect::<Vec<String>>();
    let file_path   = Path::new(&args[0]);
    let size        = file_len(file_path);
    let mut file_in = new_input_file(4096, file_path);

    let width: i32 = 512;
    let height: i32 = 256;
    let fwidth  = width  as f64;
    let fheight = height as f64;
    let fsize   = size   as f64;

    println!("Drawing fv.bmp {} by {} from {} ({} bytes)",
        width, height, file_path.display(), size);
    
    // Create blank white image
    let mut img = Image::new(width, height);
    for i in 0..width {
        for j in 0..height {
            img.put(i, j, 255, 255, 255);
        }
    }

    // Draw tick marks on the Y axis (log base 10 scale)
    let y_label_width: i32 = width / 50;
    let mut i = 1;
    if height >= 2 && width >= 2 {
        loop {
            for j in 1i32..10 {
                if (i * j as u64) < size {
                    let a = ((i * j as u64) as f64).ln();
                    let b = (fsize).ln();
                    let r = (fheight * a / b) as i32;
                    for k in 0..y_label_width {
                        img.put(k, r, -255/j, -255/j, -255/j);
                    }    
                }
            }
            if i * 10 > size { break; }
            i *= 10;
        }
    }
    
    // Darken x,y where there are matching strings at x and y (scaled) in s
    let csd    = 10.0 * fwidth * fheight / (fsize + 0.5);
    let cs     = csd as i32 + 1;
    let l2     = fheight / (2.0 + fsize).ln();
    let ilog   = Ilog::new(l2);
    let xscale = fwidth * 0.98 / fsize;
    let mut index = vec![0u32; HSIZE];
    let csd_max = csd * 32767.0;
    let rec_max = 1.0 / 32767.0;
    fastrand::seed(1);

    // Do 4 passes through file, first finding 1 byte matches (black), 
    // then 2 (red), then 4 (green), then 8 (blue).
    for i in 0i32..4 {
        let start_pass = Instant::now();
        file_in.seek(SeekFrom::Start(0)).unwrap();
        let mut h: u32 = 0;
        let mut xd: f64 = y_label_width as f64 - xscale;
        if i >= 2 {
            for i in index.iter_mut() { *i = 0; }
        }
        
        for j in 0..size {
            let c = file_in.read_byte() as u32;
            h = match i {
                0 => c + 0x10000,                   // Hash of last byte,    1st pass
                1 => (h * 256 + c) & 0xFFFF,        // Hash of last 2 bytes, 2nd pass
                2 => h * 29 * 256 + c + 1,          // Hash of last 4 bytes, 3rd pass
                _ => h * (16 * 123456789) + c + 1,  // Hash of last 8 bytes, 4th pass
            };

            let mut p = index[((h ^ (h >> 16)) as usize) & (HSIZE-1)];
            xd += xscale;

            let chksum = (h & 0xC0000000) as u32;

            if p > chksum && p < (chksum as u64 + j) as u32 {
                let x = xd as i32;
                let r = (fastrand::u16(0..32767) as f64) * rec_max;
                let y = ilog.ilog(r + (j as i64 + ((chksum as i64) - (p as i64))) as f64);

                if cs > 1 || (fastrand::u16(0..32767) as f64) < csd_max {
                    match i {
                        0 => { img.put(x, y, -cs, -cs, -cs); } // Black, 1st pass 
                        1 => { img.put(x, y, 0,   -cs, -cs); } // Red,   2nd pass
                        2 => { img.put(x, y, -cs, 0,   -cs); } // Green, 3rd pass 
                        _ => { img.put(x, y, -cs, -cs, 0  ); } // Blue,  4th pass
                    }
                }  
            }
            p = (j + chksum as u64) as u32;
            index[((h ^ (h >> 16)) as usize) & (HSIZE-1)] = p;
        }
        println!("Drew part {} of 4 in {:.2?} seconds",
            i + 1, start_pass.elapsed());
    }
    img.save_bmp("fv.bmp");
    println!("Created fv.bmp in {:.2?} seconds", 
        time.elapsed());
}


