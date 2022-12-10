from transforms3d import euler

from .sdf_utils import *


class DeepSDFOptimizer:
    def __init__(self, decoder, optimize_shape=False, lr=0.002, device="cuda"):
        self.decoder = decoder
        self.device = device

        self.optimize_shape = optimize_shape

        self.dpose = torch.tensor(
            [0, 0, 0, 1e-12, 1e-12, 1e-12],
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )

        if optimize_shape:
            self.d_latent = torch.zeros(
                (1, 256), dtype=torch.float32, requires_grad=True, device=self.device
            )
            self.optimizer = optim.Adam([self.dpose, self.d_latent], lr=0.002)
        else:
            self.d_latent = torch.zeros(
                (1, 256), dtype=torch.float32, requires_grad=False, device=self.device
            )
            self.optimizer = optim.Adam([self.dpose], lr=lr)

        self.size_est = None

        self.loss = nn.L1Loss(reduction="sum")
        self.loss = self.loss.to(self.device)

        self.ratio = 2.0

    def decode_sdf(
        self, latent_vector, points, clamp_dist=0.1, MAX_POINTS=100000, no_grad=False
    ):
        start, num_all = 0, points.shape[0]
        output_list = []
        while True:
            end = min(start + MAX_POINTS, num_all)
            if latent_vector is None:
                inputs = points[start:end]
            else:
                latent_repeat = latent_vector.expand(end - start, -1)
                inputs = torch.cat([latent_repeat, points[start:end]], 1)
            sdf_batch = self.decoder.inference(inputs)
            start = end
            if no_grad:
                sdf_batch = sdf_batch.detach()
            output_list.append(sdf_batch)
            if end == num_all:
                break
        sdf = torch.cat(output_list, 0)

        if clamp_dist != None:
            sdf = torch.clamp(sdf, -clamp_dist, clamp_dist)
        return sdf

    def compute_dist(self, d_pose, T_oc_0, ps_c, latent_vector):

        ps_o = torch.mm(
            Oplus(T_oc_0, d_pose, device=self.device), ps_c.permute(1, 0)
        ).permute(1, 0)[:, :3]

        if self.size_est is None:
            self.size_est = torch.max(torch.norm(ps_o, dim=1)).detach()
        # print('ps_o shape = ', ps_o.shape)
        # size_est = torch.max(torch.norm(ps_o, dim=1)).detach()
        size_est = self.size_est.detach()

        points_obj_norm = ps_o / size_est
        # Transform from Nocs object frame to ShapeNet object frame
        rotm_obj2shapenet = euler.euler2mat(0.0, np.pi / 2.0, 0.0)
        rotm_tensor = torch.from_numpy(rotm_obj2shapenet).float().to(self.device)
        points_obj_shapenet = torch.mm(
            rotm_tensor, points_obj_norm.permute(1, 0)
        ).permute(
            1, 0
        )  # np.dot(rotm_obj2shapenet, points_obj_norm.T).T

        dist = self.decode_sdf(latent_vector, points_obj_shapenet)

        return dist, size_est

    def eval_latent_vector(self, T_co_0, ps_c, latent_vector):
        T_oc_0 = torch.from_numpy(np.linalg.inv(T_co_0)).to(self.device)

        self.dpose.data[:3] *= 0
        self.dpose.data[3:] = self.dpose.data[3:] * 0 + 1e-12

        dist, size_est = self.compute_dist(
            self.dpose.detach(), T_oc_0, ps_c, latent_vector
        )

        return torch.mean(torch.abs(dist)).detach().cpu().numpy()

    def refine_pose(self, T_co_0, ps_c, latent_vector, steps=100, shape_only=False):
        # input T_co_0: 4x4
        #       ps_c:   nx4

        T_oc_0 = torch.from_numpy(np.linalg.inv(T_co_0)).to(self.device)

        self.dpose.data[:3] *= 0
        self.dpose.data[3:] = self.dpose.data[3:] * 0 + 1e-12
        self.d_latent.data *= 0

        if shape_only:
            self.dpose.requires_grad = False
            init_lr = 0.002
            decreased_by = 10
            adjust_lr_every = int(steps / 2)

        for i in range(steps):
            if shape_only:
                self.adjust_learning_rate(
                    init_lr, self.optimizer, i, decreased_by, adjust_lr_every
                )
            self.optimizer.zero_grad()

            dist, size_est = self.compute_dist(
                self.dpose, T_oc_0, ps_c, latent_vector + self.d_latent
            )
            dist_target = torch.zeros_like(dist)
            dist_target = dist_target.to(self.device)

            loss = self.loss(dist, dist_target) + 1e-4 * torch.mean(
                (latent_vector + self.d_latent).pow(2)
            )
            loss.backward()

            if not shape_only:
                self.dpose.grad[3:] *= 2
            self.optimizer.step()

            # print('step: {}, loss = {}'.format(i + 1, loss.data.cpu().item()))

        # print('step: {}, loss = {}'.format(i + 1, loss.data.cpu().item()))
        T_oc_opt = Oplus(T_oc_0, self.dpose, device=self.device)
        T_co_opt = np.linalg.inv(T_oc_opt.cpu().detach().numpy())

        latent_vector_opt = (latent_vector + self.d_latent).detach()

        dist = torch.mean(torch.abs(dist)).detach().cpu().numpy()

        return (
            T_co_opt,
            dist,
            latent_vector_opt,
            size_est.detach().cpu().numpy(),
            loss.data.cpu().item(),
        )

    def adjust_learning_rate(
        self, initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
