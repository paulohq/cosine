import struct

fd_out = open('bin_file', 'wb')

#[2-byte ID][4-byte value]

id = 0
val = id

for i in range(50):
    entry = struct.pack('<HI', id, val)
    id += 1
    val = id

    fd_out.write(entry)
    fd_out.flush()

fd_out.close()
