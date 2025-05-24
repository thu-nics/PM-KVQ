import torch


def pack_16bit_tensor(w_q):
    w_q_org_shape = w_q.shape
    w_q = w_q.to(torch.int16).reshape(-1, 1)
    w_int16 = torch.zeros(w_q.shape[0], 1, dtype=torch.int16, device=w_q.device)

    w_int16[:, 0] |= w_q[:, 0]

    new_shape = w_q_org_shape[:-1] + (w_q_org_shape[-1] * 16 // 16,)
    return w_int16.reshape(new_shape)


def pack_8bit_tensor(w_q):
    w_q_org_shape = w_q.shape
    w_q = w_q.to(torch.int16).reshape(-1, 2)
    w_int8 = torch.zeros(w_q.shape[0], 1, dtype=torch.int16, device=w_q.device)

    w_int8[:, 0] |= w_q[:, 0] << 8
    w_int8[:, 0] |= w_q[:, 1]

    new_shape = w_q_org_shape[:-1] + (w_q_org_shape[-1] * 8 // 16,)
    return w_int8.reshape(new_shape)


def pack_4bit_tensor(w_q):
    w_q_org_shape = w_q.shape
    w_q = w_q.to(torch.int16).reshape(-1, 4)
    w_int4 = torch.zeros(w_q.shape[0], 1, dtype=torch.int16, device=w_q.device)

    w_int4[:, 0] |= w_q[:, 0] << 12
    w_int4[:, 0] |= w_q[:, 1] << 8
    w_int4[:, 0] |= w_q[:, 2] << 4
    w_int4[:, 0] |= w_q[:, 3]

    new_shape = w_q_org_shape[:-1] + (w_q_org_shape[-1] * 4 // 16,)
    return w_int4.reshape(new_shape)


def pack_2bit_tensor(w_q):
    w_q_org_shape = w_q.shape
    w_q = w_q.to(torch.int16).reshape(-1, 8)
    w_int2 = torch.zeros(w_q.shape[0], 1, dtype=torch.int16, device=w_q.device)

    w_int2[:, 0] |= w_q[:, 0] << 14
    w_int2[:, 0] |= w_q[:, 1] << 12
    w_int2[:, 0] |= w_q[:, 2] << 10
    w_int2[:, 0] |= w_q[:, 3] << 8
    w_int2[:, 0] |= w_q[:, 4] << 6
    w_int2[:, 0] |= w_q[:, 5] << 4
    w_int2[:, 0] |= w_q[:, 6] << 2
    w_int2[:, 0] |= w_q[:, 7]

    new_shape = w_q_org_shape[:-1] + (w_q_org_shape[-1] * 2 // 16,)
    return w_int2.reshape(new_shape)


def unpack_16bit_tensor(w_int16):
    w_int16_org_shape = w_int16.shape
    new_shape = w_int16_org_shape[:-1] + (w_int16_org_shape[-1] * 16 // 16,)
    w_int16 = w_int16.reshape(-1, 1)
    w_q = torch.zeros(w_int16.shape[0], 1, dtype=torch.int16, device=w_int16.device)

    w_q[:, 0] = w_int16[:, 0]

    return w_q.reshape(new_shape)


def unpack_8bit_tensor(w_int8):
    w_int8_org_shape = w_int8.shape
    new_shape = w_int8_org_shape[:-1] + (w_int8_org_shape[-1] * 16 // 8,)
    w_int8 = w_int8.reshape(-1, 1)
    w_q = torch.zeros(w_int8.shape[0], 2, dtype=torch.int16, device=w_int8.device)

    w_q[:, 0] = w_int8[:, 0] >> 8
    w_q[:, 1] = (w_int8[:, 0] << 8) >> 8

    return w_q.reshape(new_shape)


def unpack_4bit_tensor(w_int4):
    w_int4_org_shape = w_int4.shape
    new_shape = w_int4_org_shape[:-1] + (w_int4_org_shape[-1] * 16 // 4,)
    w_int4 = w_int4.reshape(-1, 1)
    w_q = torch.zeros(w_int4.shape[0], 4, dtype=torch.int16, device=w_int4.device)

    w_q[:, 0] = w_int4[:, 0] >> 12
    w_q[:, 1] = (w_int4[:, 0] << 4) >> 12
    w_q[:, 2] = (w_int4[:, 0] << 8) >> 12
    w_q[:, 3] = (w_int4[:, 0] << 12) >> 12

    return w_q.reshape(new_shape)


def unpack_2bit_tensor(w_int2):
    w_int2_org_shape = w_int2.shape
    new_shape = w_int2_org_shape[:-1] + (w_int2_org_shape[-1] * 16 // 2,)
    w_int2 = w_int2.reshape(-1, 1)
    w_q = torch.zeros(w_int2.shape[0], 8, dtype=torch.int16, device=w_int2.device)

    w_q[:, 0] = w_int2[:, 0] >> 14
    w_q[:, 1] = (w_int2[:, 0] << 2) >> 14
    w_q[:, 2] = (w_int2[:, 0] << 4) >> 14
    w_q[:, 3] = (w_int2[:, 0] << 6) >> 14
    w_q[:, 4] = (w_int2[:, 0] << 8) >> 14
    w_q[:, 5] = (w_int2[:, 0] << 10) >> 14
    w_q[:, 6] = (w_int2[:, 0] << 12) >> 14
    w_q[:, 7] = (w_int2[:, 0] << 14) >> 14

    return w_q.reshape(new_shape)


pack_funcs = {16: pack_16bit_tensor, 8: pack_8bit_tensor, 4: pack_4bit_tensor, 2: pack_2bit_tensor}
unpack_funcs = {16: unpack_16bit_tensor, 8: unpack_8bit_tensor, 4: unpack_4bit_tensor, 2: unpack_2bit_tensor}
