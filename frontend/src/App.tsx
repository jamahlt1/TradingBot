import React from 'react';
import { AppBar, Toolbar, Typography, Drawer, List, ListItem, ListItemText, CssBaseline, Box } from '@mui/material';

const drawerWidth = 220;
const sections = [
  'Dashboard',
  'Strategies',
  'Accounts',
  'Trades',
  'Analytics',
  'Settings',
];

function App() {
  const [selected, setSelected] = React.useState('Dashboard');

  return (
    <Box sx={{ display: 'flex' }}>
      <CssBaseline />
      <AppBar position="fixed" sx={{ zIndex: 1201 }}>
        <Toolbar>
          <Typography variant="h6" noWrap component="div">
            Trading Bot Dashboard
          </Typography>
        </Toolbar>
      </AppBar>
      <Drawer
        variant="permanent"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          [`& .MuiDrawer-paper`]: { width: drawerWidth, boxSizing: 'border-box' },
        }}
      >
        <Toolbar />
        <Box sx={{ overflow: 'auto' }}>
          <List>
            {sections.map((text) => (
              <ListItem button key={text} selected={selected === text} onClick={() => setSelected(text)}>
                <ListItemText primary={text} />
              </ListItem>
            ))}
          </List>
        </Box>
      </Drawer>
      <Box component="main" sx={{ flexGrow: 1, bgcolor: 'background.default', p: 3, ml: `${drawerWidth}px` }}>
        <Toolbar />
        <Typography variant="h4" gutterBottom>
          {selected}
        </Typography>
        {/* TODO: Render section content here */}
      </Box>
    </Box>
  );
}

export default App;
