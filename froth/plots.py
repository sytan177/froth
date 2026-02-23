import numpy as np

def mask_pos_slice(pos_i, tar_pos, thickness=0.005):
    """
    Mask particles within a slice around target position.
    
    Parameters
    ----------
    pos_i : np.ndarray
        Particle coordinates along axis
    tar_pos : float
        Target position for slice center
    thickness : float, default=0.005
        Half-thickness of the slice
    
    Returns
    -------
    mask : np.ndarray
        Boolean mask for particles within slice
    """
    return np.abs(pos_i - tar_pos) < thickness

def mask_pos_range(pos_i, pos_start, pos_end):
    """
    Mask particles within a coordinate range.
    
    Parameters
    ----------
    pos_i : np.ndarray
        Particle coordinates along axis
    pos_start : float
        Range start
    pos_end : float
        Range end
    
    Returns
    -------
    mask : np.ndarray
        Boolean mask for particles in [pos_start, pos_end]
    """
    return (pos_i > pos_start) & (pos_i < pos_end)

'''def mask_line(gas_pos, x_pos = None, y_pos = None, z_pos = None, thickness = 0.005, plot_range = 1.):
    import numpy as np
    # Plot the line along a specific direction given a position at the plane of the rest two axes
    if x_pos is None:
        mask = (np.abs(gas_pos[:, 1] - y_pos) <= thickness) & (np.abs(gas_pos[:, 2] - z_pos) <= thickness) 
    elif y_pos is None:
        mask = (np.abs(gas_pos[:, 0] - x_pos) <= thickness) & (np.abs(gas_pos[:, 2] - z_pos) <= thickness) 
    elif z_pos is None:
        mask = (np.abs(gas_pos[:, 0] - x_pos) <= thickness) & (np.abs(gas_pos[:, 1] - y_pos) <= thickness) 
    else:
        raise ValueError("`Two of x_pos, y_pos, or z_pos must be specified")
    return mask'''

def mask_line(gas_pos, x_pos=None, y_pos=None, z_pos=None, thickness=0.005):
    """
    Mask particles along a line (1D slice through 3D space).
    
    Specify exactly two positions to define the perpendicular plane;
    masks particles near the line extending along the third axis.
    
    Parameters
    ----------
    gas_pos : np.ndarray of shape (n, 3)
        Particle positions [x, y, z]
    x_pos : float, optional
        Fixed x-coordinate
    y_pos : float, optional
        Fixed y-coordinate
    z_pos : float, optional
        Fixed z-coordinate
    thickness : float, default=0.005
        Half-thickness of slice around the line
    
    Returns
    -------
    mask : np.ndarray
        Boolean mask for particles near the line
    
    Examples
    --------
    >>> mask = mask_line(pos, y_pos=0.5, z_pos=0.2)  # line along x-axis
    """
    specified = sum([x_pos is not None, y_pos is not None, z_pos is not None])
    if specified != 2:
        raise ValueError("Exactly two of x_pos, y_pos, z_pos must be specified")
    
    if x_pos is None:
        mask = (np.abs(gas_pos[:, 1] - y_pos) <= thickness) & (np.abs(gas_pos[:, 2] - z_pos) <= thickness)
    elif y_pos is None:
        mask = (np.abs(gas_pos[:, 0] - x_pos) <= thickness) & (np.abs(gas_pos[:, 2] - z_pos) <= thickness)
    else:  # z_pos is None
        mask = (np.abs(gas_pos[:, 0] - x_pos) <= thickness) & (np.abs(gas_pos[:, 1] - y_pos) <= thickness)
    return mask

def plot_gas_scatter_2d(gas_pos_masked, masked_quantities, plot_axes=[0, 2], 
                        cmaps=None, vmin=None, vmax=None, marker_size=0.005,
                        figsize=None, dpi=150, subplot_kw=None, **scatter_kw):
    """
    Plot 2D gas particle distributions.
    
    Parameters
    ----------
    gas_pos_masked : np.ndarray of shape (n, 3)
        Masked particle positions
    masked_quantities : list of np.ndarray
        Quantities to color particles (one per subplot)
    plot_axes : list of int, default=[0, 2]
        Axes to plot [x_axis, y_axis] from [0=X, 1=Y, 2=Z]
    cmaps : list of str, optional
        Colormaps for each subplot
    vmin, vmax : list of float, optional
        Color limits for each subplot
    marker_size : float, default=0.005
        Marker size (s parameter for scatter)
    figsize : tuple, optional
        Figure size (width, height). Default: (6*n_plots, 6)
    dpi : int, default=150
        Figure DPI
    subplot_kw : dict, optional
        Additional kwargs passed to plt.subplots()
    **scatter_kw
        Additional kwargs passed to ax.scatter() (applied to all subplots)
    
    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    
    n_plots = len(masked_quantities)
    figsize = figsize or (6 * n_plots, 6)
    subplot_kw = subplot_kw or {}
    
    fig, ax = plt.subplots(1, n_plots, figsize=figsize, dpi=dpi, **subplot_kw)
    if n_plots == 1:
        ax = [ax]
    
    axis_names = ['x [kpc]', 'y [kpc]', 'z [kpc]']
    
    for i in range(n_plots):
        quantity = masked_quantities[i]
        scatter_kwargs = {
            'c': quantity,
            's': marker_size,
            **scatter_kw}
        
        if cmaps is not None and i < len(cmaps):
            scatter_kwargs['cmap'] = cmaps[i]
        if vmin is not None and i < len(vmin):
            scatter_kwargs['vmin'] = vmin[i]
        if vmax is not None and i < len(vmax):
            scatter_kwargs['vmax'] = vmax[i]
        
        ax[i].scatter(gas_pos_masked[:, plot_axes[0]], 
                     gas_pos_masked[:, plot_axes[1]], 
                     **scatter_kwargs)
        ax[i].set_aspect('equal', adjustable='box')
        ax[i].set_xlabel(axis_names[plot_axes[0]])
    
    ax[0].set_ylabel(axis_names[plot_axes[1]])
    plt.tight_layout()
    return fig, ax

def plot_gas_scatter_subplots(gas_pos, masks=[], planes=[], quantities=[], 
                               plot_w_size=3, plot_h_size=6,
                               cmaps=None, cbars=None, marker_size=0.005, 
                               titles=None, cbar_labels=None, 
                               vmin=None, vmax=None, 
                               sharex=True, sharey=True,
                               subplot_kw=None, **scatter_kw):
    """
    Create multiple 2D gas scatter subplots with flexible masking and plane selection.
    
    Parameters
    ----------
    gas_pos : np.ndarray of shape (n, 3)
        Particle positions [x, y, z]
    masks : list of np.ndarray
        Boolean masks for each subplot
    planes : list of str
        Projection planes for each subplot: 'xy', 'yz', or 'xz'
    quantities : list of np.ndarray
        Quantities to color particles in each subplot
    plot_w_size : float, default=3
        Width per subplot in inches
    plot_h_size : float, default=6
        Height of figure in inches
    cmaps : list of str, optional
        Colormaps for each subplot
    cbars : list of bool, optional
        Whether to show colorbar for each subplot
    marker_size : float, default=0.005
        Marker size for scatter plot
    titles : list of str, optional
        Subplot titles
    cbar_labels : list of str, optional
        Colorbar labels
    vmin, vmax : list of float, optional
        Color limits for each subplot
    sharex, sharey : bool, default=True
        Share axes across subplots
    subplot_kw : dict, optional
        Additional kwargs for plt.subplots()
    **scatter_kw
        Additional kwargs for ax.scatter()
    
    Returns
    -------
    fig, axes : matplotlib figure and axes
    
    Raises
    ------
    ValueError
        If masks, planes, and quantities have different lengths
    """
    import matplotlib.pyplot as plt
    
    n_plots = len(quantities)
    
    if not (len(masks) == len(planes) == len(quantities)):
        raise ValueError(f"masks ({len(masks)}), planes ({len(planes)}), "
                        f"and quantities ({len(quantities)}) must have the same length")
    
    subplot_kw = subplot_kw or {}
    fig, ax = plt.subplots(1, n_plots, 
                          figsize=(plot_w_size*n_plots, plot_h_size), 
                          dpi=200, sharex=sharex, sharey=sharey,
                          **subplot_kw)
    if n_plots == 1:
        ax = [ax]
    
    plane_config = {
        'xy': (0, 1, 'X [kpc]', 'Y [kpc]'),
        'yz': (1, 2, 'Y [kpc]', 'Z [kpc]'),
        'xz': (0, 2, 'X [kpc]', 'Z [kpc]')
    }
    
    for i in range(n_plots):
        quantity = quantities[i]
        plane = planes[i].lower()
        mask = masks[i]
        
        if plane not in plane_config:
            raise ValueError(f"Invalid plane '{plane}'. Must be one of: xy, yz, xz")
        
        ax_ind0, ax_ind1, xlabel, ylabel = plane_config[plane]
        
        scatter_kwargs = {
            'c': quantity[mask],
            's': marker_size,
            **scatter_kw
        }
        if cmaps is not None and i < len(cmaps):
            scatter_kwargs['cmap'] = cmaps[i]
        if vmin is not None and i < len(vmin):
            scatter_kwargs['vmin'] = vmin[i]
        if vmax is not None and i < len(vmax):
            scatter_kwargs['vmax'] = vmax[i]
        
        im = ax[i].scatter(gas_pos[mask, ax_ind0], 
                          gas_pos[mask, ax_ind1], 
                          **scatter_kwargs)
        
        if cbars is not None and i < len(cbars) and cbars[i]:
            cbar = fig.colorbar(im, ax=ax[i])
            if cbar_labels is not None and i < len(cbar_labels):
                cbar.set_label(cbar_labels[i])
        
        if titles is not None and i < len(titles):
            ax[i].set_title(titles[i])
        
        ax[i].set_xlabel(xlabel)
        ax[i].set_ylabel(ylabel)
        ax[i].set_aspect('equal', adjustable='box')
    
    # Use shared labels if all subplots use same plane
    all_same_plane = len(set(planes)) == 1
    if all_same_plane:
        plane = planes[0].lower()
        _, _, xlabel, ylabel = plane_config[plane]
        for a in ax:
            a.set_xlabel('')
            a.set_ylabel('')
        fig.supxlabel(xlabel)
        fig.supylabel(ylabel)
    
    plt.tight_layout()
    return fig, ax

def plot_gas_line_profiles(gas_pos, masks=[], axis=[], quantities=[], 
                          color_quantities=[], ylabel=None, 
                          plot_w_size=8, plot_h_size=2.5, 
                          cmaps=None, marker_size=0.01, 
                          titles=None, vmin=None, vmax=None, 
                          cbar_labels=None, subplot_kw=None, **scatter_kw):
    """
    Plot 1D profiles of gas quantities along coordinate axes.
    
    Parameters
    ----------
    gas_pos : np.ndarray of shape (n, 3)
        Particle positions [x, y, z]
    masks : list of np.ndarray
        Boolean masks for each subplot
    axis : list of str
        Axis for each profile: 'x', 'y', or 'z'
    quantities : list of np.ndarray
        Quantities to plot on y-axis for each subplot
    color_quantities : list of np.ndarray, optional
        Quantities to color points. If not provided, uses solid color
    ylabel : list of str, optional
        Y-axis labels. Default: 'log Quantity'
    plot_w_size : float, default=8
        Figure width in inches
    plot_h_size : float, default=2.5
        Height per subplot in inches
    cmaps : list of str, optional
        Colormaps for each subplot
    marker_size : float, default=0.01
        Marker size for scatter plot
    titles : list of str, optional
        Subplot titles
    vmin, vmax : list of float, optional
        Color limits for each subplot
    cbar_labels : list of str, optional
        Colorbar labels
    subplot_kw : dict, optional
        Additional kwargs for plt.subplots()
    **scatter_kw
        Additional kwargs for ax.scatter()
    
    Returns
    -------
    fig, axes : matplotlib figure and axes
    
    Raises
    ------
    ValueError
        If masks, axis, and quantities have different lengths
    """
    import matplotlib.pyplot as plt
    
    n_plots = len(quantities)
    
    if not (len(masks) == len(axis) == len(quantities)):
        raise ValueError(f"masks ({len(masks)}), axis ({len(axis)}), "
                        f"and quantities ({len(quantities)}) must have the same length")
    
    subplot_kw = subplot_kw or {}
    fig, ax = plt.subplots(n_plots, 1, 
                          figsize=(plot_w_size, plot_h_size*n_plots), 
                          dpi=150, sharex=True, sharey=False,
                          **subplot_kw)
    if n_plots == 1:
        ax = [ax]
    
    axis_config = {
        'x': (0, 'X [kpc]'),
        'y': (1, 'Y [kpc]'),
        'z': (2, 'Z [kpc]')
    }
    
    for i in range(n_plots):
        quantity = quantities[i]
        dirc = axis[i].lower()
        mask = masks[i]
        
        if dirc not in axis_config:
            raise ValueError(f"Invalid axis '{dirc}'. Must be one of: x, y, z")
        
        ax_ind, xlabel = axis_config[dirc]
        
        # Handle color
        if color_quantities is not None and i < len(color_quantities):
            color = color_quantities[i][mask]
        else:
            color = 'purple'
        
        scatter_kwargs = {
            'c': color,
            's': marker_size,
            **scatter_kw
        }
        if cmaps is not None and i < len(cmaps):
            scatter_kwargs['cmap'] = cmaps[i]
        if vmin is not None and i < len(vmin):
            scatter_kwargs['vmin'] = vmin[i]
        if vmax is not None and i < len(vmax):
            scatter_kwargs['vmax'] = vmax[i]
        
        im = ax[i].scatter(gas_pos[mask, ax_ind], quantity[mask], **scatter_kwargs)
        
        # Add colorbar if using color quantity
        if color_quantities is not None and i < len(color_quantities):
            cbar = fig.colorbar(im, ax=ax[i])
            if cbar_labels is not None and i < len(cbar_labels):
                cbar.set_label(cbar_labels[i])
        
        ax[i].set_xlabel(xlabel)
        
        if titles is not None and i < len(titles):
            ax[i].set_title(titles[i])
        
        if ylabel is not None and i < len(ylabel):
            ax[i].set_ylabel(ylabel[i])
        else:
            ax[i].set_ylabel('log Quantity')
    
    plt.tight_layout()
    return fig, ax

def plot_gas_phase_space(quantity_x, quantity_y, mask=None, labels=None, 
                        bin_number=500, figsize=(8, 6), dpi=150, 
                        cmap='viridis', norm=None, hist2d_kw=None, **fig_kw):
    """
    Plot 2D histogram (phase space diagram) of two gas quantities.
    
    Parameters
    ----------
    quantity_x : np.ndarray
        Values for x-axis
    quantity_y : np.ndarray
        Values for y-axis
    mask : np.ndarray, optional
        Boolean mask to apply to both quantities
    labels : list of str, optional
        [xlabel, ylabel] for axis labels
    bin_number : int, default=500
        Number of bins for 2D histogram
    figsize : tuple, default=(8, 6)
        Figure size (width, height) in inches
    dpi : int, default=150
        Figure DPI
    cmap : str, default='viridis'
        Colormap
    norm : matplotlib.colors.Normalize, optional
        Normalization for colormap (default: LogNorm())
    hist2d_kw : dict, optional
        Additional kwargs for ax.hist2d()
    **fig_kw
        Additional kwargs for plt.subplots()
    
    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, **fig_kw)
    
    hist2d_kw = hist2d_kw or {}
    norm = norm if norm is not None else LogNorm()
    
    x_data = quantity_x[mask] if mask is not None else quantity_x
    y_data = quantity_y[mask] if mask is not None else quantity_y
    
    h = ax.hist2d(x_data, y_data, bins=bin_number, norm=norm, 
                  cmap=cmap, **hist2d_kw)
    
    plt.colorbar(h[3], ax=ax)
    
    if labels is not None and len(labels) >= 2:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
    
    return fig, ax