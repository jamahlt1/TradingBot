import React from 'react';
import { Typography, Paper, Box } from '@mui/material';

const Settings: React.FC = () => (
  <Box>
    <Typography variant="h5" gutterBottom>
      Settings
    </Typography>
    <Paper sx={{ p: 2, mb: 2 }}>
      <Typography>User and system settings UI will appear here.</Typography>
    </Paper>
  </Box>
);

export default Settings;