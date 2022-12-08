class FLAGS:
    obj_c = 6

    img_size = 256

    feat_pcl = 1286
    feat_global_pcl = 512
    feat_seman = 32
    R_c = 4
    Ts_c = 6
    feat_face = 768

    face_recon_c = 6 * 5
    gcn_n_num = 10
    gcn_sup_num = 7

    use_global_feat_for_ts = 0
    use_point_conf_for_vote = 1
    use_seman_feat = 0
    support_points = 516
    random_points = 512

    train = False
    eval_coord = 0
    eval_recon = 1
